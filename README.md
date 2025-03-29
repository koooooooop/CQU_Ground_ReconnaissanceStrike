项目结构如下
CQUGRS/
  ├── config/
  │    ├── camera_intrinsics.yaml    # (可选)相机标定/内参
  │    └── flight_params.yaml        # (可选)飞控相关配置
  ├── datasets/
  │    ├── train/
  │    │    ├── images/
  │    │    └── labels/
  │    ├── valid/
  │    │    ├── images/
  │    │    └── labels/
  │    └── data.yaml                 # YOLO 训练配置
  ├── scripts/
  │    ├── data_collection/
  │    │    └── capture_images.py    # 数据采集脚本(可选)
  │    ├── training/
  │    │    └── train_yolo.py       # 训练脚本
  │    └── inference/
  │         ├── main.py             # 主程序：推理 + 旋转 + OCR
  │         └── utils.py            # (可选) 工具函数
  ├── tmp_storage/                   # (可选)临时存储拍摄原图
  ├── ready/                         # (可选)旋正后图像等
  ├── weights/
  │    └── best.pt                  # 训练好的模型权重
  ├── README.md
  └── requirements.txt               # (可选)所需依赖清单

