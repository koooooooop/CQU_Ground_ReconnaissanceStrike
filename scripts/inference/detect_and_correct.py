# detect_and_correct.py
import cv2
import numpy as np
from ultralytics import YOLO
import math

def rotate_image(image, angle):
    """根据旋转角度旋转图像"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (width, height))
    return rotated_image

def detect_and_rotate(model, image_path):
    """检测图像并根据箭头框旋转图像"""
    # 读取图像
    img = cv2.imread(image_path)
    
    # 使用 YOLO 模型进行推理
    results = model.predict(img)
    
    # 提取检测框 (假设只有一个框：箭头框)
    boxes = results[0].boxes
    if len(boxes) > 0:
        # 取第一个检测框
        x1, y1, x2, y2 = boxes[0].xyxy[0].tolist()
        # 计算中心点和旋转角度
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        angle = math.degrees(math.atan2(center_y - y1, center_x - x1))  # 假设框是指向顶部，计算旋转角度

        # 旋转图像
        rotated_image = rotate_image(img, angle)
        return rotated_image
    else:
        print("没有检测到框")
        return img

def save_rotated_images():
    # 初始化 YOLOv8n 模型
    model = YOLO("weights/best.pt")
    input_folder = "tmp_storage"
    output_folder = "ready"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历tmp_storage文件夹中的所有图像
    for image_name in os.listdir(input_folder):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(input_folder, image_name)
            rotated_image = detect_and_rotate(model, image_path)

            # 保存旋转后的图像
            rotated_image_path = os.path.join(output_folder, f"rotated_{image_name}")
            cv2.imwrite(rotated_image_path, rotated_image)
            print(f"保存旋转后的图像：{rotated_image_path}")

if __name__ == "__main__":
    save_rotated_images()
