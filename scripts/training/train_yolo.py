import os
from ultralytics import YOLO

def train_yolo(
    data_yaml='my_dataset/data.yaml',
    model_arch='yolov8n.pt',
    epochs=50,
    batch_size=16,
    project='runs',
    name='digit_detector'
):
    """
    使用 YOLOv8n 训练检测模型的示例函数。
    参数：
      data_yaml  : data.yaml 文件的路径
      model_arch : 初始模型权重，默认 yolov8n.pt
      epochs     : 训练轮数
      batch_size : 批量大小，根据显存适当调整
      project    : 训练结果保存的根目录
      name       : 本次训练保存的子目录名称
    """
    # 初始化模型
    model = YOLO(model_arch)

    # 开始训练
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        project=project,
        name=name
    )

if __name__ == "__main__":
    # 示例用法
    # 已将 data.yaml 放在 my_dataset/ 目录下
    data_file = 'datasets/data.yaml'

    # 调用训练函数
    train_yolo(
        data_yaml=data_file,
        model_arch='yolov8n.pt',  # YOLOv8n 轻量版
        epochs=50,
        batch_size=16,
        project='runs',          # 默认存储路径
        name='cone_digit_detector'
    )
