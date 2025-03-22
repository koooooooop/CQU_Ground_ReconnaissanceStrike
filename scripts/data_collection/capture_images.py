import cv2
import numpy as np
import os

# 替换为你的 RTSP 地址
rtsp_url = "rtsp://192.168.144.25:8554/main.264"

def detect_and_mark_red(frame):
    """
    在给定的图像frame中识别红色区域，并用矩形框标记所有红色块。
    返回标记后的图像。
    """

    # 将BGR图像转换为HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围（示例：分两段）
    # 低色调红
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    # 高色调红
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 生成掩膜
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # 一些形态学操作，去除噪点或填充空洞（可根据需要调整）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上绘制矩形框
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤掉太小的区域，避免噪点
        if w < 5 or h < 5:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame

def main():
    # 1. 创建 VideoCapture 对象并打开 RTSP 视频流
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("无法打开 RTSP 流，请检查地址或网络设置。")
        return

    # 2. 设置保存目录，若不存在则自动创建
    save_dir = r"E:\desktop\CQUGRS\tmp_storage"
    os.makedirs(save_dir, exist_ok=True)

    # 帧计数器 & 已保存图片计数器
    frame_count = 0
    saved_img_count = 0

    # 标记是否已经显示过 "Captured Frame" 窗口
    window_shown = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧，或视频结束。")
            break

        # 显示原始视频流
        cv2.imshow("RTSP Stream", frame)

        frame_count += 1

        # 每 15 帧执行一次检测并保存
        if frame_count % 15 == 0:
            # 若窗口已存在，则先关闭，避免重复销毁报错
            if window_shown:
                try:
                    cv2.destroyWindow("Captured Frame")
                except cv2.error:
                    pass

            # ---- 关键部分：检测并标记红色区域 ----
            marked_frame = detect_and_mark_red(frame.copy())

            # 展示带有标记（红色矩形框）的图像
            cv2.imshow("Captured Frame", marked_frame)
            window_shown = True

            # 将带有标记的图像保存到指定文件夹
            filename = f"frame_{saved_img_count}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, marked_frame)
            print(f"已保存图像: {filepath}")

            saved_img_count += 1

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 退出时，释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()