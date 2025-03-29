# yolo_utils.py

from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="weights/best.pt", conf_thres=0.25):
        """
        model_path: YOLO 模型权重文件 (best.pt)
        conf_thres: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        """
        对单帧图像进行检测，返回检测结果。
        结果形式：
        [
          {
            'box': (x1, y1, x2, y2),
            'conf': 0.9,
            'class_id': 0
          },
          ...
        ]
        """
        results = self.model.predict(frame, conf=self.conf_thres)
        boxes = []
        if len(results) > 0:
            # 只取第一张图的检测结果
            result = results[0]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                boxes.append({
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'class_id': cls_id
                })
        return boxes
