import cv2
from easy_ViTPose import VitInference
import os

base_path = '/ballet_positions/testing/dataset_train'
files = os.listdir(base_path)
target_files = []
num = 0
for f in files:
    if ('.jpg' in f) or ('.jpeg' in f) or ('.JPG' in f):
        target_files.append(os.path.join(base_path, f))
for x in target_files:
    num += 1

    img = cv2.imread(x)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model_path = '/ballet_positions/pose detection/easy_ViTPose_try/easy_ViTPose/vitpose-l-coco.pth'
    yolo_path = '/ballet_positions/pose detection/easy_ViTPose_try/easy_ViTPose/yolov8s.pt'

    model = VitInference(model_path, yolo_path, model_name='l', yolo_size=320, is_video=False, device='cpu')

    keypoints = model.inference(img)

    img = model.draw(show_yolo=True)  # Returns RGB image with drawings
    # cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); cv2.waitKey(0)
    out_path = '/ballet_positions/testing/easy_ViTPose_results'
    output_path = os.path.splitext(out_path)[0] + f"/_pose_detected{num}.jpg"
    cv2.imwrite(output_path, img)