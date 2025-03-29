# main.py
import cv2
import math
import easyocr
import numpy as np
from ultralytics import YOLO

def compute_rotation_angle(box):
    """
    简化示例：根据检测框 box=[x1,y1,x2,y2] 计算旋转角度 (度数)。
    若你有 apex/digit_box 两类，需要根据 apex->digit_box连线算角度。
    这里仅以框自身方向为示例(同left->right)。
    """
    x1, y1, x2, y2 = box
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def rotate_and_crop(frame, box):
    """
    1. 计算 box 的旋转角 angle
    2. 以 box 中心为基点旋转整图
    3. 截取 box 区域
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    angle = compute_rotation_angle(box)

    # 构造仿射变换矩阵
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    h, w = frame.shape[:2]
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderValue=(255,255,255))

    # 在旋转后的图像中，按原 box 位置裁剪
    x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
    cropped = rotated[y1_i:y2_i, x1_i:x2_i].copy()

    return cropped

def main():
    # 加载 YOLOv8 模型 (推理)
    model = YOLO("weights/best.pt")   # 你的训练好模型

    # 初始化 OCR
    reader = easyocr.Reader(['en'], gpu=False)  # Jetson Nano可尝试gpu=True

    # 打开摄像头(或RTSP)
    cap = cv2.VideoCapture(0)  # 0表示USB或CSI摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("帧获取失败")
            break

        # 1. YOLO 推理
        results = model.predict(source=frame, conf=0.25)
        boxes = results[0].boxes
        # 如果只关心一个目标，可取最高置信度
        if len(boxes) > 0:
            best_conf = 0
            best_box = None
            for b in boxes:
                conf = float(b.conf[0].item())
                if conf > best_conf:
                    best_conf = conf
                    best_box = b.xyxy[0].tolist()

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                # 2. 旋转并裁剪
                cropped = rotate_and_crop(frame, [x1, y1, x2, y2])

                # 3. OCR 识别
                ocr_result = reader.readtext(cropped)
                recognized_texts = [res[1] for res in ocr_result]
                recognized_text = " ".join(recognized_texts)

                # 打印或保存结果
                print(f"OCR 识别结果：{recognized_text}")

                # 画框+文字
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, recognized_text, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # 显示画面(如不需要UI，可注释掉)
        cv2.imshow("Jetson Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
