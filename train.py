import argparse
import sys
import subprocess

import torch
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.utils.collate_fn import DetectionCollateFN
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed
from super_gradients.training import Trainer
from super_gradients.training import models

from utils import parse_yaml, create_dataloader, create_train_params, plot_samples


def train(
    config = './config.yaml',
    augment = './augment.yaml',
    model_name = 'yolo_nas_s',   # ['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l']
    ckpt = './ckpt_best.pth', 
    img_size = 640,
    ch = 3,
    cache = False,
    silent = False,
    ema = False,
    ignore_empty = False,
    pretrain = False,
    gpu = False,
    batch_size = 16,
    workers = 0,
    epochs = 300,
    optimizer = 'AdamW',     # ['SGD', 'Adam', 'AdamW']
    name = 'exp',
    resume = '',
    plot = False,
    wandb = False
):
    # activate w&b tracking training
    if wandb:
        try:
            import wandb
        except ModuleNotFoundError:
            print('The wandb module is NOT installed, Installing...')

            python = sys.executable
            subprocess.check_call(
                ['pip', 'install', 'wandb'],
                stdout=subprocess.DEVNULL
            )
            print("Successful installation!")
        finally:
            import wandb

    # set device
    setup_device(num_gpus=-1) if torch.cuda.is_available() and gpu else setup_device(device="cpu")

    # get dataset and hyperparameters path 
    config = parse_yaml(config)
    augment = parse_yaml(augment)
    num_classes = len(config['classes'])
    class_names = list(config['classes'].values())

    # create train and val dataloader
    train_loader = create_dataloader(
        config,
        augment,
        collate_fn=DetectionCollateFN(), 
        worker_init_fn=worker_init_reset_seed, 
        img_size=img_size, 
        cache=cache, 
        ignore_empty_annotations=ignore_empty,
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True, 
        num_workers=workers, 
        persistent_workers= workers > 0,
        mode="train"
    )

    val_loader = create_dataloader(
        config,
        collate_fn=DetectionCollateFN(), 
        worker_init_fn=worker_init_reset_seed, 
        img_size=img_size, 
        cache=cache, 
        ignore_empty_annotations=ignore_empty,
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True, 
        num_workers=workers, 
        persistent_workers=workers > 0,
        mode="val"
    )
    
    if plot:
        plot_samples(train_loader.dataset, num_samples=4, save_path=f'./transformed_samples.png')

    # get model
    if pretrain and ckpt:
        model = models.get(model_name, num_classes=num_classes, checkpoint_path=ckpt, num_input_channels=ch, load_backbone=True)
    elif pretrain:
        model = models.get(model_name, num_classes=num_classes, pretrained_weights="coco", num_input_channels=ch)
    elif ckpt:
        model = models.get(model_name, num_classes=num_classes, checkpoint_path=ckpt, num_input_channels=ch)
    else:
        model = models.get(model_name, num_classes=num_classes, num_input_channels=ch)

    train_params = create_train_params(
        class_names, 
        num_classes, 
        optimizer=optimizer, 
        ema=ema, 
        silent=silent, 
        resume_path=resume, 
        epochs=epochs
    )

    # training
    trainer = Trainer(experiment_name=name, ckpt_root_dir="./runs/checkpoints/")
    trainer.train(model=model, training_params=train_params, train_loader=train_loader, valid_loader=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./config.yaml', help='config.yaml path')
    parser.add_argument('--augment', type=str, default='./augment.yaml', help='augment.yaml path')
    parser.add_argument('--model_name', '-m', type=str, choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'], default='yolo_nas_s', help='model')
    parser.add_argument('--ckpt', type=str, help='initial weights path')
    parser.add_argument('--img-size', '--img', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--ch', type=int, default=3, help='image channel')
    parser.add_argument('--cache', action="store_true", help='whether to cache images or not')
    parser.add_argument('--silent', action="store_true", help='hide training details')
    parser.add_argument('--ema', action="store_true", help='activate exponential moving average')
    parser.add_argument('--ignore-empty', '--ignore', action="store_true", help='ignore empty annotations')
    parser.add_argument('--pretrain', action="store_true", help='load pretrain model')
    parser.add_argument('--gpu', action="store_true", help='activate gpu or cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='AdamW', help='optimizer')
    parser.add_argument('--name', type=str, default='exp', help='save to project/name')
    parser.add_argument('--resume', type=str, help='direct path to a specific checkpoint file (.pth) to resume training.')
    parser.add_argument('--plot', action="store_true", help='plot transformed data')
    parser.add_argument('--wandb', action="store_true", help='visualize YOLO-NAS predictions')

    opt = parser.parse_args()
    print(opt)

    train(**vars(opt))