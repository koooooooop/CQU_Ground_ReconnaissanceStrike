# yolo_detector.py

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# 如果需要OCR或旋转操作，可从外部导入函数
# from .ocr_utils import rotate_image, ocr_recognition

def main():
    # 1. 加载训练好的 YOLO 模型
    model = YOLO("weights/best.pt")  # 确保该文件存在

    # 2. 监控目录
    watch_dir = "tmp_storage"
    processed_files = set()  # 用于记录已处理的文件名

    while True:
        # 获取当前目录所有文件
        files = sorted(os.listdir(watch_dir))

        for f in files:
            if not f.lower().endswith(".jpg"):
                continue
            if f in processed_files:
                continue  # 已处理过

            filepath = os.path.join(watch_dir, f)
            # 读取图像
            img = cv2.imread(filepath)
            if img is None:
                continue

            # 3. YOLO推理
            results = model.predict(img, conf=0.25)
            boxes = results[0].boxes
            print(f"[yolo_detector] 检测到 {len(boxes)} 个目标。")

            # 4. 对检测框进行后处理(旋转, OCR等)
            #   下面仅简单演示画框, 如果需要OCR, 需import easyocr
            if len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

                # 也可以在这里进行“旋转并裁剪, OCR识别”等操作
                # cropped = rotate_image(...)
                # text = ocr_recognition(cropped)
                # print("识别结果:", text)

            # 5. 可视化结果保存到 ready/ 或直接覆盖
            save_path = os.path.join("ready", f)  # ready文件夹
            os.makedirs("ready", exist_ok=True)
            cv2.imwrite(save_path, img)
            print(f"[yolo_detector] 识别结果已保存到: {save_path}")

            processed_files.add(f)  # 标记已处理

        # 间隔一会再扫目录, 避免空转
        time.sleep(2)

if __name__ == "__main__":
    main()
