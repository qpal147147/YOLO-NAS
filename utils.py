import yaml
import matplotlib.pyplot as plt

import cv2
import numpy as np
from super_gradients.training.datasets.detection_datasets.yolo_format_detection import YoloDarknetFormatDetectionDataset
from super_gradients.training.transforms.transforms import DetectionRandomAffine, DetectionHSV, DetectionHorizontalFlip, DetectionVerticalFlip, DetectionMosaic, DetectionMixup, DetectionPaddedRescale, DetectionStandardize, DetectionTargetsFormatTransform, DetectionPadToSize
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from torch.utils.data import DataLoader

def parse_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        try:
            info = yaml.safe_load(f)
            return info
        except yaml.YAMLError as exc:
            print(exc)
            return None

def create_dataset(config, augment=None, img_size=640, cache=False, ignore_empty_annotations=False, mode="train"):
    if mode == "train":
        if augment == None: raise FileNotFoundError("augment.yaml not found !")

        train_set = YoloDarknetFormatDetectionDataset(
            data_dir = config["root"],
            images_dir = config["train"],
            labels_dir = config["train"].replace("images", "labels"),
            input_dim=(img_size, img_size),
            classes = list(config['classes'].values()),
            show_all_warnings = True,
            ignore_empty_annotations=ignore_empty_annotations,
            cache = cache,
            cache_dir=config["root"],
            transforms=[
                DetectionMosaic(input_dim=img_size, prob=augment["mosaic"]),
                DetectionRandomAffine(degrees=augment["randomAffine_degree"], scales=augment["randomAffine_scales"], shear=augment["randomAffine_shear"], target_size=img_size, filter_box_candidates=True, border_value=augment["randomAffine_border"]),
                DetectionMixup(input_dim=img_size, mixup_scale=augment["mixup_scale"], prob=augment["mixup"]),
                DetectionHSV(prob=augment["hsv"], hgain=augment["hsv_h"], sgain=augment["hsv_s"], vgain=augment["hsv_v"]),
                DetectionHorizontalFlip(prob=augment["fliplr"]),
                DetectionVerticalFlip(prob=augment["flipud"]),
                DetectionPaddedRescale(input_dim=img_size),
                DetectionStandardize(max_value=255),
                DetectionTargetsFormatTransform(input_dim=img_size, output_format="LABEL_CXCYWH"),
            ]
        )
        return train_set
        
    elif mode == "val":
        val_set = YoloDarknetFormatDetectionDataset(
            data_dir = config["root"],
            images_dir = config["val"],
            labels_dir = config["val"].replace("images", "labels"),
            input_dim=(img_size, img_size),
            classes = list(config['classes'].values()),
            show_all_warnings = True,
            ignore_empty_annotations=ignore_empty_annotations,
            cache = cache,
            cache_dir=config["root"],
            transforms=[
                DetectionPaddedRescale(input_dim=img_size),
                DetectionStandardize(max_value=255),
                DetectionTargetsFormatTransform(input_dim=img_size, output_format="LABEL_CXCYWH"),
            ],
        )
        return val_set
    
    elif mode == "test":
        test_set = YoloDarknetFormatDetectionDataset(
            data_dir = config["root"],
            images_dir = config["test"],
            labels_dir = config["test"].replace("images", "labels"),
            input_dim=(img_size, img_size),
            classes = list(config['classes'].values()),
            show_all_warnings = True,
            ignore_empty_annotations=ignore_empty_annotations,
            cache = cache,
            cache_dir=config["root"],
            transforms=[
                DetectionPaddedRescale(input_dim=img_size),
                DetectionStandardize(max_value=255),
                DetectionTargetsFormatTransform(input_dim=img_size, output_format="LABEL_CXCYWH"),
            ],
        )
        return test_set
    
    else:
        raise Exception("which one do you choose 'train', 'val' or 'test'?")

def create_dataloader(config, augment=None, collate_fn=None, worker_init_fn=None, img_size=640, cache=False, ignore_empty_annotations=False, batch_size=16, shuffle=True, pin_memory=True, num_workers=8, persistent_workers=True, mode="train"):
    if mode == "train":
        train_set = create_dataset(config, augment, img_size, cache=cache, ignore_empty_annotations=ignore_empty_annotations, mode="train")
        return DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn, 
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

    elif mode == "val":
        val_set = create_dataset(config, img_size, cache=cache, ignore_empty_annotations=ignore_empty_annotations, mode="val")
        return DataLoader(
            val_set, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn, 
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

    elif mode == "test":
        test_set = create_dataset(config, img_size=img_size, cache=cache, ignore_empty_annotations=ignore_empty_annotations, mode="test")
        return DataLoader(
            test_set, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn, 
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
    
    else:
        raise Exception("which one do you choose 'train', 'val' or 'test'?")
    
def create_train_params(class_names, num_classes, optimizer="AdamW", ema=False, silent=False, resume_path="", epochs=300):
    train_params = {
        'silent_mode': silent,
        'resume_path': resume_path,
        "average_best_models": False,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": optimizer,
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": ema,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": epochs,
        "mixed_precision": True,
        "loss": PPYoloELoss(use_static_assigner=False, num_classes=num_classes, reg_max=16),
        "valid_metrics_list": [
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
        ],
        "metric_to_watch": 'mAP@0.50:0.95',
        "sg_logger": "wandb_sg_logger",
        "sg_logger_params": {
            "project_name": "YOLO_NAS", # W&B project name
            "save_checkpoints_remote": False,
            "save_tensorboard_remote": True,
            "save_logs_remote": True
        }
    }

    return train_params

def create_processing_params(class_names:list ,img_size=640, iou=0.65, conf=0.5):
    params = {
        'class_names': class_names, 
        'image_processor': {
            'ComposeProcessing': {
                'processings': [
                    'ReverseImageChannels', 
                    {
                        'DetectionLongestMaxSizeRescale': {
                            'output_shape': (img_size, img_size)
                        }
                    }, 
                    {
                        'DetectionLongestMaxSizeRescale': {
                            'output_shape': (img_size, img_size)
                        }
                    }, 
                    {
                        'DetectionBottomRightPadding': {
                            'output_shape': (img_size, img_size), 
                            'pad_value': 114
                        }
                    }, 
                    {
                        'ImagePermute': {
                            'permutation': (2, 0, 1)
                        }
                    }, 
                    {
                        'StandardizeImage': {
                            'max_value': 255.0
                        }
                    }
                ]
            }
        }, 
        'iou': iou, 
        'conf': conf
    }

    return params


def transform_coordinates(original_coords, original_size, new_size, pad_info=(0, 0, 0, 0)):
    x1, y1, x2, y2 = original_coords
    orig_height, orig_width = original_size

    new_height, new_width = new_size
    pad_top, pad_bot, pad_left, pad_right = pad_info

    # subtract the padding information to calculate the actual position.
    orig_width -= (pad_left + pad_right)
    orig_height -= (pad_top + pad_bot)
    x1 -= pad_left
    y1 -= pad_top
    x2 -= pad_left
    y2 -= pad_top

    # calculate scaling ratio
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height

    # apply the scaling.
    new_x1 = int(x1 * scale_x)
    new_y1 = int(y1 * scale_y)
    new_x2 = int(x2 * scale_x)
    new_y2 = int(y2 * scale_y)
    
    return new_x1, new_y1, new_x2, new_y2


def resize_and_pad(image, target_size, pad_value=114):
    h, w = image.shape[:2]
    
    # interpolation method
    if h > target_size or w > target_size: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = target_size
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (target_size-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = target_size
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (target_size-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = target_size, target_size
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(image.shape)==3 and not isinstance(pad_value, (list, tuple, np.ndarray)): # color image but only one color provided
        pad_value = [pad_value]*3
    
    # scale and pad
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_value)

    return scaled_img, (pad_top, pad_bot, pad_left, pad_right)


def plot_samples(dataset, num_samples=4, save_path=''):
    images = [np.transpose(dataset.__getitem__(i)[0], (1,2,0)) for i in range(num_samples)]
    
    fig, axes = plt.subplots(2, 2, figsize=(6,6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    info = parse_yaml('./augment.yaml')
    print(info)