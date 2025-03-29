# camera_inference.py

import cv2
import time
from yolo_utils import YOLODetector
from ocr_utils import rotate_and_crop, ocr_recognition

# 替换成你的 RTSP/USB 设备索引
RTSP_URL = "rtsp://192.168.144.25:8554/main.264"
# RTSP_URL = 0  # 如果是USB摄像头，或者CSI摄像头

def main():
    # 1. 初始化YOLO检测器
    detector = YOLODetector(model_path="weights/best.pt", conf_thres=0.25)

    # 2. 打开摄像头
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("无法打开摄像头/RTSP流")
        return

    # 3. 实时循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("获取帧失败")
            time.sleep(0.5)
            continue

        # 4. YOLO检测
        boxes = detector.detect(frame)
        # 可以根据需求处理多个或一个目标，这里仅示例遍历
        for box_info in boxes:
            x1, y1, x2, y2 = box_info['box']
            conf = box_info['conf']
            # class_id = box_info['class_id']

            # 5. 在画面上画检测框(可选)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            # 显示置信度
            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

            # 6. 旋转裁剪 + OCR
            cropped = rotate_and_crop(frame, (x1, y1, x2, y2))
            text = ocr_recognition(cropped)
            print(f"[INFO] OCR结果: {text}")

            # 也可把OCR结果显示在主画面
            cv2.putText(frame, text, (int(x1), int(y2)+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # 7. 显示最终画面(若在无人机无屏幕场景，可以不用显示)
        cv2.imshow("Jetson Nano RealTime", frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 27=ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
