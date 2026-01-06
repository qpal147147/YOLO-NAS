import argparse

import torch
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils.collate_fn import DetectionCollateFN
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed

from utils import parse_yaml, create_dataloader


def val(
    config = './config.yaml',
    augment = './augment.yaml',
    ckpt = './ckpt_best.pth',
    img_size = 640,
    task = 'val',    # ["train", "val", "test"]
    model_name = 'yolo_nas_s', # ['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l']
    ch = 3,
    gpu = False,
    cache = False,
    ignore_empty = False,
    batch_size = 16,
    workers = 0,
    name = 'val'
):
    # set device
    setup_device(num_gpus=-1) if torch.cuda.is_available() and gpu else setup_device(device="cpu")
    
    # get dataset path 
    config = parse_yaml(config)
    augment = parse_yaml(augment)
    num_classes = len(config['classes'])
    class_names = list(config['classes'].values())
    
    # create dataloader
    if task == "train":
        dataloader = create_dataloader(
            config,
            augment,
            collate_fn=DetectionCollateFN(),
            worker_init_fn=worker_init_reset_seed,
            img_size = img_size, 
            cache=cache, 
            ignore_empty_annotations=ignore_empty,
            batch_size=batch_size, 
            shuffle=False, 
            num_workers= workers,
            persistent_workers=workers > 0,
            mode=task
        )
    elif task == "val":
        dataloader = create_dataloader(
            config,
            collate_fn=DetectionCollateFN(),
            worker_init_fn=worker_init_reset_seed,
            img_size = img_size, 
            cache=cache, 
            ignore_empty_annotations=ignore_empty,
            batch_size=batch_size, 
            shuffle=False, 
            num_workers= workers,
            persistent_workers=workers > 0,
            mode=task
        )
    elif task == "test":
        dataloader = create_dataloader(
            config,
            collate_fn=DetectionCollateFN(),
            worker_init_fn=worker_init_reset_seed,
            img_size = img_size, 
            cache=cache, 
            ignore_empty_annotations=ignore_empty,
            batch_size=batch_size, 
            shuffle=False, 
            num_workers= workers,
            persistent_workers=workers > 0,
            mode=task
        )

    # get model
    model = models.get(model_name, num_classes=num_classes, checkpoint_path=ckpt, num_input_channels=ch)
    
    # evaluate model
    trainer = Trainer(experiment_name=name, ckpt_root_dir="./runs")
    regular_metrics = trainer.test(
        model=model, 
        test_loader=dataloader,
        test_metrics_list=[
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=num_classes,
                normalize_targets=True,
                class_names=class_names,
                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
            ), 
            DetectionMetrics_050_095(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=num_classes,
                normalize_targets=True,
                class_names=class_names,
                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7)
            )
        ]
    )

    print(regular_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='./config.yaml', help='config.yaml path')
    parser.add_argument('--augment', type=str, default='./augment.yaml', help='augment.yaml path')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/ckpt_best.pth', help='checkpoints.pth path')
    parser.add_argument('--img-size', '--img', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--task', type=str, choices=["train", "val", "test"], default='val', help='verification directory')
    parser.add_argument('--model_name', '-m', type=str, choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'], default='yolo_nas_s', help='model')
    parser.add_argument('--ch', type=int, default=3, help='image channel')
    parser.add_argument('--gpu', action="store_true", help='activate gpu or cpu')
    parser.add_argument('--cache', action="store_true", help='whether to cache images or not')
    parser.add_argument('--ignore-empty', '--ignore', action="store_true", help='ignore empty annotations')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers')
    parser.add_argument('--name', type=str, default='val', help='save to project/name')
    
    opt = parser.parse_args()
    print(opt)

    val(**vars(opt))