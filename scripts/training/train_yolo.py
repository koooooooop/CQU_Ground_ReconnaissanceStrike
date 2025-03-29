# train_yolo.py
from ultralytics import YOLO

def main():
    # 使用 ultralytics 提供的预训练 YOLOv8n 模型
    model = YOLO("yolov8n.pt")

    # 开始训练
    model.train(
        data="datasets/data.yaml",  # 数据配置文件
        epochs=50,                  # 训练轮数
        batch=16,                   # batch大小, 按显存调节
        name="digit_detector",      # 输出的项目名称
        project="runs/detect"       # 结果存放目录
    )

if __name__ == "__main__":
    main()
