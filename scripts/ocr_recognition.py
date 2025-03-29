# ocr_recognition.py
import easyocr
import os
import cv2

def ocr_recognition(image_path):
    """使用 OCR 识别图像中的数字"""
    reader = easyocr.Reader(['en'])  # 识别英文数字
    result = reader.readtext(image_path)
    
    # 提取识别的文本（通常是数字）
    text = ""
    for res in result:
        text += res[1] + " "
    return text.strip()

def process_ready_images():
    input_folder = "ready"
    
    for image_name in os.listdir(input_folder):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(input_folder, image_name)
            text = ocr_recognition(image_path)
            print(f"识别结果：{image_name} -> {text}")

if __name__ == "__main__":
    process_ready_images()
