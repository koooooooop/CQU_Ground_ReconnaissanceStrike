# ocr_utils.py

import cv2
import math
import easyocr

# 初始化全局 OCR 对象（避免每次都初始化）
reader = easyocr.Reader(['en'], gpu=False)

def compute_rotation_angle(x1, y1, x2, y2):
    """
    简单示例：假设 box 左上角->右下角 为旋转方向
    或者只以宽的方向算倾斜角
    """
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def rotate_and_crop(frame, box):
    """
    box = (x1,y1,x2,y2).
    1) 根据 box 算 angle
    2) 以 box 中心做仿射变换
    3) 裁剪
    """
    x1, y1, x2, y2 = box
    angle = compute_rotation_angle(x1, y1, x2, y2)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    h, w = frame.shape[:2]
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

    # 裁剪
    x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
    cropped = rotated[y1_i:y2_i, x1_i:x2_i].copy()

    return cropped

def ocr_recognition(image):
    """
    对图像执行 OCR，并返回识别到的文本字符串
    """
    result = reader.readtext(image)
    texts = [res[1] for res in result]
    return " ".join(texts)
