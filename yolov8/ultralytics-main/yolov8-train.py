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
