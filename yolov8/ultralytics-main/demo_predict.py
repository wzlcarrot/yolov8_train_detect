from ultralytics import YOLO

yolo = YOLO("/home/wzlcarrot/project/yolov8/ultralytics-main/best.pt",task="detect")

# conf表示的是置信度
result = yolo(source="/home/wzlcarrot/project/yolov8/ultralytics-main/ultralytics/assets/960.jpg", save=True)
 