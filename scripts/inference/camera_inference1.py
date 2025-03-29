# camera_inference.py

import cv2
import time
from yolo_utils import YOLODetector
from ocr_utils import rotate_and_crop, ocr_recognition

def main():
    # 1. 初始化YOLO检测器(使用你的 best.pt)
    detector = YOLODetector(model_path="weights/best.pt", conf_thres=0.25)

    # 2. 打开本地摄像头：索引 = 0
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("无法打开本地摄像头。请检查索引或设备。")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("获取帧失败。")
            time.sleep(0.5)
            continue

        # 3. YOLO 检测
        boxes = detector.detect(frame)

        # 4. 对检测结果进行画框、旋转、OCR等
        for box_info in boxes:
            x1, y1, x2, y2 = box_info['box']
            conf = box_info['conf']
            # 可根据 box_info['class_id'] 判断类别

            # 在图像上绘制检测框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

            # 旋转并裁剪
            cropped = rotate_and_crop(frame, (x1, y1, x2, y2))

            # OCR识别
            text = ocr_recognition(cropped)
            print(f"[INFO] OCR识别结果: {text}")

            # 将OCR结果显示在检测框下方
            cv2.putText(frame, text, (int(x1), int(y2)+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # 显示实时画面
        cv2.imshow("Local Camera Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
