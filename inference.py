import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import numpy as np
import cv2
import onnxruntime
import torch
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.datasets.data_formats.bbox_formats.normalized_cxcywh import xyxy_to_normalized_cxcywh
from super_gradients.training import models

from utils import parse_yaml, create_processing_params, resize_and_pad, transform_coordinates


def inference(
    config = './config.yaml',
    source = './images',
    img_size = 640,
    conf_thres = 0.25,
    iou_thres = 0.45,
    model_name = 'yolo_nas_s', # ['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l']
    ckpt = './ckpt_best.pth',
    ch = 3,
    gpu = False,
    line_thickness = 2,
    hide_conf = False,
    save_txt = False,
    crop = False
):
    # set device
    setup_device(num_gpus=-1) if torch.cuda.is_available() and gpu else setup_device(device="cpu")

    # get source data
    ROOT = Path(source)
    if ROOT.is_file():
        files = [ROOT]

    elif ROOT.is_dir():
        files = list(ROOT.glob("*.jpg"))
    
    # create model
    if Path(ckpt).suffix == ".pth":
        config = parse_yaml(config)
        num_classes = len(config['classes'])
        model = models.get(model_name, num_classes=num_classes, checkpoint_path=ckpt, num_input_channels=ch)
        model.set_dataset_processing_params(**create_processing_params(config['classes'], img_size, iou=iou_thres, conf=conf_thres))
    elif Path(ckpt).suffix == ".onnx":
        model = onnxruntime.InferenceSession(ckpt, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    else:
        raise FileNotFoundError

    # create save directory
    save_root = f"./runs/inference/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    Path(save_root).mkdir(parents=True, exist_ok=True)

    # inference
    for i, path in enumerate(tqdm(files)):
        if Path(ckpt).suffix == ".pth":
            out = model.predict(str(path), fuse_model=False, skip_image_resizing=False)

            # prediction info
            image = out.image
            xyxy_bboxes = out.prediction.bboxes_xyxy
            xywh_bboxes = xyxy_to_normalized_cxcywh(xyxy_bboxes, (image.shape[0], image.shape[1]))
            confidences = out.prediction.confidence
            labels = out.prediction.labels
            class_names = out.class_names
        
        elif Path(ckpt).suffix == ".onnx":
            detect_inputs = [o.name for o in model.get_inputs()]
            detect_outputs = [o.name for o in model.get_outputs()]

            input_image = cv2.imdecode(np.fromfile(file=path, dtype=np.uint8), cv2.IMREAD_COLOR)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            padded_img, pad_info = resize_and_pad(input_image, img_size)
            image_bchw = np.transpose(padded_img[np.newaxis,:,:,:], (0, 3, 1, 2))

            # detector predict
            predictions = model.run(detect_outputs, {detect_inputs[0]: image_bchw})

            # prediction info
            image = input_image
            xyxy_bboxes, xywh_bboxes, confidences, labels, class_names = [], [], [], [], []
            for batch, x1, y1, x2, y2, confidence, class_index in predictions[0]:
                x1, y1, x2, y2 = transform_coordinates((x1, y1, x2, y2), (img_size, img_size), input_image.shape[:2], pad_info)
                xyxy_bboxes.append([x1, y1, x2, y2])
                xywh_bboxes.append(xyxy_to_normalized_cxcywh(np.array([[x1, y1, x2, y2]]), (image.shape[0], image.shape[1])))
                confidences.append(confidence)
                labels.append(class_index)

        if crop:
            image_dir = f"{save_root}/{'image'}"
            Path(image_dir).mkdir(parents=True, exist_ok=True)

            for idx, xyxy in enumerate(xyxy_bboxes):
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                new_y1, new_y2 = max(y1-10, 0), min(y2+10, image.shape[0])
                new_x1, new_x2 = max(x1-10, 0), min(x2+10, image.shape[1])
                crop_image = image[new_y1:new_y2, new_x1:new_x2]

                filename = f"{image_dir}/{path.stem}_{idx}.jpg"
                cv2.imwrite(filename, crop_image)

        if save_txt:
            txt_dir = f"{save_root}/{'txt'}"
            Path(txt_dir).mkdir(parents=True, exist_ok=True)

            with open(f"{txt_dir}/{path.stem}.txt", "w") as f:
                for label, xywh , xyxy in zip(labels, xywh_bboxes, xyxy_bboxes):
                    line = f"{label} {str(xywh[0])} {str(xywh[1])} {str(xywh[2])} {str(xywh[3])}\n"
                    f.write(line)

        # save prediction
        drawed_image = image.copy()
        for xyxy, label, conf in zip(xyxy_bboxes, labels, confidences):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cv2.rectangle(drawed_image, (x1, y1), (x2, y2), color=(255,0,0), thickness=line_thickness)
            cv2.putText(drawed_image, str(int(label)), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), line_thickness)
            if not hide_conf:
                cv2.putText(drawed_image, f"{conf:.3f}", (x1+30, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), line_thickness)
            cv2.imwrite(f"{save_root}/{path.name}", drawed_image)
        # out.save(output_path=f"{save_root}/{path.name}", box_thickness=line_thickness, show_confidence=not hide_conf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='./config.yaml', help='config.yaml path')
    parser.add_argument('--source', type=str, default='./images', help='file or dir')
    parser.add_argument('--img-size', '--img', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument("--conf-thres", '--conf', type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", '--iou', type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument('--model-name', '-m', type=str, choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'], default='yolo_nas_s', help='model')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/ckpt_best.pth', help='checkpoints.pth or .onnx path')
    parser.add_argument('--ch', type=int, default=3, help='image channel')
    parser.add_argument('--gpu', action="store_true", help='activate gpu or cpu')
    parser.add_argument('--line-thickness', '--line', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--save-txt', '--txt', action="store_true", help='save results to *.txt')
    parser.add_argument('--crop', action="store_true", help='crop detected image')

    opt = parser.parse_args()
    print(opt)

    inference(**vars(opt))