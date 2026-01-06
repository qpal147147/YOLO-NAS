import argparse
from pathlib import Path

import numpy as np
import cv2
import onnxruntime
from super_gradients.conversion import DetectionOutputFormatMode
from super_gradients.conversion import ExportTargetBackend
from super_gradients.training import models
    
from utils import parse_yaml, create_processing_params, resize_and_pad, transform_coordinates

def test_onnx(onnx, img_size, image_path):
    input_image = cv2.imdecode(np.fromfile(file=image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    padded_img, pad_info = resize_and_pad(input_image, img_size)
    image_bchw = np.transpose(padded_img[np.newaxis,:,:,:], (0, 3, 1, 2))

    detector = onnxruntime.InferenceSession(onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    detect_inputs = [o.name for o in detector.get_inputs()]
    detect_outputs = [o.name for o in detector.get_outputs()]

    predictions = detector.run(detect_outputs, {detect_inputs[0]: image_bchw})
    for batch, x1, y1, x2, y2, confidence, class_index in predictions[0]:
        x1, y1, x2, y2 = transform_coordinates((x1, y1, x2, y2), (img_size, img_size), input_image.shape[:2], pad_info)
        cv2.rectangle(input_image, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)

    cv2.imshow("test", input_image)
    cv2.waitKey(0)

def export(
    config = './config.yaml',
    img_size = 640,
    conf_thres = 0.25,
    iou_thres = 0.45,
    model_name = 'yolo_nas_s', # ['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l']
    ckpt = './ckpt_best.pth',
    ch = 3,
):
    
    # create model
    config = parse_yaml(config)
    num_classes = len(config['classes'])
    model = models.get(model_name, num_classes=num_classes, checkpoint_path=ckpt, num_input_channels=ch)
    model.set_dataset_processing_params(**create_processing_params(config['classes'], img_size, iou=iou_thres, conf=conf_thres))

    # export onnx
    output = Path(ckpt).with_suffix(".onnx")
    export_result = model.export(
        str(output), 
        preprocessing=True,
        confidence_threshold = conf_thres,
        nms_threshold = iou_thres,
        num_pre_nms_predictions = 1000,
        output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
        engine=ExportTargetBackend.ONNXRUNTIME
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./config.yaml', help='config.yaml path')
    parser.add_argument('--img-size', '--img', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument("--conf-thres", '--conf', type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", '--iou', type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument('--model-name', '-m', type=str, choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'], default='yolo_nas_s', help='model')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/ckpt_best.pth', help='checkpoints.pth path')
    parser.add_argument('--ch', type=int, default=3, help='image channel')


    opt = parser.parse_args()
    print(opt)

    export(**vars(opt))

    # test_onnx("./ckpt_best.onnx", opt.img_size, "./test.jpg")
