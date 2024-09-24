## 简介

主要的任务：

1. 搞清yolov8的网络结构图。
2. 熟悉源码。
3. 熟悉yolov8训练和推理过程。



## 训练和推理过程

### 1. 训练过程

（1）准备数据集，并且还要打上标签，通过视频分帧，分成了许多张图片，并且打上标签。

（2）直接编写yolov8-train.py文件

```python
from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")
model = YOLO("yolov8n.pt")  # load a pretrained model 不使用预训练权重，就注释这一行即可

# train
model.train(data='yolo-bvn.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=2,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                amp=False, # close amp
                project='project/yolov8/ultralytics-main/runs/train',
                name='exp',
                )
```

（3）编写yolo-bvn.yaml配置文件

```yaml
path: /home/wzlcarrot/project/yolov8/ultralytics-main/datasets/bvn
train: images/train
val: images/val
test:

names:
  0: daitu
  1: mingren
```

（4）编写demo_predict.py文件

```yaml
from ultralytics import YOLO

yolo = YOLO("/home/wzlcarrot/project/yolov8/ultralytics-main/best.pt",task="detect")

# conf表示的是置信度
result = yolo(source="/home/wzlcarrot/project/yolov8/ultralytics-main/ultralytics/assets/960.jpg", save=True)
```

（4）执行train.py文件，结果在runs/train/exp目录下。

### 2. 推理过程

（1）在runs/train/exp/weights/目录下可以找到best.pt。

（2）直接执行predict.py文件，结果在runs/predict/目录下。
