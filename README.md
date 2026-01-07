# YOLO-NAS

> [!NOTE]
> This project is derived from the [original SuperGradients repository](https://github.com/Deci-AI/super-gradients) and is simplified for easier use. It focuses **only on basic object detection tasks**, including training, validation, testing, and ONNX export.
>
>If you’re looking for the full feature set and official tutorials, please visit the original repository or check out the archived [documentation](https://web.archive.org/web/20240328172223/https://docs.deci.ai/super-gradients/latest/documentation/source/welcome.html).
> 
> To reduce the learning curve, the training workflow is designed to be as close as possible to [YOLOv5](https://github.com/ultralytics/yolov5). This allows you to focus on your own tasks instead of learning a new training pipeline.
>
> If you’d like to improve this project or add more features, feel free to fork it :fork_and_knife: or contribute :sparkles:

## Requirements
1. Create Environment
   ```bash
   # 3.9 <= python <= 3.10
   conda create -n yolonas python=3.10
   activate yolonas
   ```

2. Install Dependencies
   ```bash
   # If a GPU is available, it is recommended to install PyTorch with CUDA support and onnxruntime-gpu according to your environment.
   # Pytorch>=2.1.0
   # onnxruntime-gpu==1.15.0
   pip install -r requirement.txt
   ```

## Data Preparation
   Use the **YOLO** annotation format: 
   ```txt
   <object-class> <x_center> <y_center> <width> <height>
   ```
   You can refer to the official [YOLOv5](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/) guide for details.  
   **Images with empty annotation files can be used as negative samples during training.**

   The final dataset should have the following structure:
   ```bash
   datasets
   └── project_name  # Dataset root
      ├── images
      │   ├── train
      │   │   ├── *.jpg(or others format)
      │   │   ├── 0001.jpg
      │   │   └── ...
      │   ├── val
      │   │   └── *.jpg(or others format)
      │   └── test
      │       └── *.jpg(or others format)
      └── labels
          ├── train
          │   ├── *.txt
          │   ├── 0001.txt
          │   └── ...
          ├── val
          │   └── *.txt
          └── test/
              └── *.txt
   ```

## Models
   You can select your model by referring to the table below.[^1]

   |   Model    | Model Name | Dataset | Resolution | mAP<sup>val<br>0.5:0.95 | :arrow_down: |
   |:----------:|:----------:|:-------:|:----------:|:-----------------------:|:--------:|
   | YOLO-NAS S | yolo_nas_s |   COCO  |   640x640  |  47.5(FP16) 47.03(INT8) | [Download](https://github.com/qpal147147/YOLO-NAS/releases/download/v1.0.0/yolo_nas_s_coco.pth) |
   | YOLO-NAS M | yolo_nas_m |   COCO  |   640x640  |  51.55(FP16) 51.0(INT8) | [Download](https://github.com/qpal147147/YOLO-NAS/releases/download/v1.0.0/yolo_nas_m_coco.pth) |
   | YOLO-NAS L | yolo_nas_l |   COCO  |   640x640  |  52.22(FP16) 52.1(INT8) | [Download](https://github.com/qpal147147/YOLO-NAS/releases/download/v1.0.0/yolo_nas_l_coco.pth) |

   [^1]: https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/model_zoo.md#pretrained-object-detection-pytorch-checkpoints

## Usage
   1. Set your dataset path and define the class names.
      ```yaml
      # config.yaml
      root: '/dataset/PASCAL_VOC2007'
      train: 'images/train'
      val: 'images/val'
      test: 'images/test'
      
      classes:
         0: aeroplane
         1: bicycle
         2: bird
         3: ...
      ```

2. Modify the `super-gradients` package to use a custom URL for the pre-trained models.
   ```python
   # ../super_gradients/training/utils/checkpoint_utils.py
   ...
   if url.startswith("file://") or os.path.exists(url):
      pretrained_state_dict = torch.load(url.replace("file://", ""), map_location="cpu")
    else:
      # unique_filename = url.split("https://sghub.deci.ai/models/")[1].replace("/", "_").replace(" ", "_")
      unique_filename = url.split("https://sg-hub-nv.s3.amazonaws.com/models/")[1].replace("/", "_").replace(" ", "_")
      map_location = torch.device("cpu")
   ...
   ```

   ```python
   # ../super_gradients/training/pretrained_models.py

   # Replace all occurrences of "https://sghub.deci.ai/models/" with "https://sg-hub-nv.s3.amazonaws.com/models/".
   MODEL_URLS = {
      "regnetY800_imagenet": "https://sg-hub-nv.s3.amazonaws.com/models/regnetY800_imagenet.pth",
      "regnetY600_imagenet": "https://sg-hub-nv.s3.amazonaws.com/models/regnetY600_imagenet.pth",
      "...": "...",
   }
   ```
### Training
   * Use a pretrained model
     ```bash
     python train.py -m yolo_nas_s --pretrain --img 640 --ch 3 --gpu --batch-size 32 --epochs 100
     ```
   * If you don't want to download the model from the internet, you can use `--ckpt` to specify the model path.
      ```bash
      python train.py -m yolo_nas_s --ckpt yolo_nas_s_coco.pth --img 640 --ch 3 --gpu --batch-size 32 --epochs 100
      ```
   * Resume Training
      ```bash
      python train.py -m yolo_nas_s --img 640 --ch 3 --gpu --batch-size 32 --epochs 100 --resume runs/checkpoints/exp/RUN_20260101_010101_000001/ckpt_latest.pth
      ```
    
> [!TIP]
> 1. You can use [Weights & Biases (wandb)](https://wandb.ai/) to monitor the model status during training in real time by enabling the `--wandb` flag.
> 2. If you want to modify the image augmentation parameters, please adjust `augment.yaml`.
> 3. To speed up training, you can use `--cache` to load the training data into memory and increase the `--workers` value to improve data loading performance.
> 4. If you find the training logs too distracting, you can disable them using the `--silent` flag.
> 5. If you do not want any negative samples included in the dataset, you can use the `--ignore-empty` flag.
> 6. If you want to change the project name used during training, please use the `--name` flag and set it to your preferred name.
> 7. If you want to visualize the augmented images, use the `--plot` flag to save the images.
    
### Validation
  ```bash
  python val.py -m yolo_nas_s --ckpt ckpt_best.pth --img 640 --ch 3 --gpu --batch-size 32
  ```

### Inference
  ```bash
  python inference.py --source test.jpg -m yolo_nas_s --ckpt ckpt_best.pth --img 640 --ch 3 --gpu
  # or
  python inference.py --source ./test -m yolo_nas_s --ckpt ckpt_best.onnx --img 640 --ch 3 --gpu
  ```

> [!TIP]
> 1. If you find that there are too many or too few bounding boxes, you can adjust the `--conf` and `--iou` thresholds.
> 2. If the bounding box lines appear too thick, try adjusting `--line-thickness`.
> 3. If the confidence scores are too cluttered, you can hide them using the `--hide-conf` flag.
> 4. If you want to save the detected objects as additional training data, you can use `--save-txt` and `--crop` to save them as `txt` and `jpg` files (all bounding boxes are expanded by 10px).

### Export ONNX
  ```bash
  python export.py -m yolo_nas_s --ckpt ckpt_best.pth --img 640 --ch 3  
  ```

## License
In accordance with the license of [super-gradients](https://github.com/Deci-AI/super-gradients), this project is also released under the Apache-2.0 license.

> [!IMPORTANT]
> Only the use of [pretrained weights](https://github.com/Deci-AI/super-gradients/issues/894#issuecomment-1534118271) is subject to non-commercial restrictions.  
> Therefore, if you require commercial use, please use randomly initialized weights instead.
