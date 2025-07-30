from ultralytics import YOLO
import cv2
import numpy as np

# 加载模型
model = YOLO("/home/xy/adas_project/yolo/yolo11n.pt")

# 图片路径
img_path = "/home/xy/adas_project/yolo/imag.jpg"

# 加载图片（用于OpenCV绘制）
img = cv2.imread(img_path)

# 推理
results = model(img_path)
result = results[0]  # 取第一个（只有一张图）

# 获取检测框、类别、置信度
boxes = result.boxes

if boxes is not None and len(boxes) > 0:
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
    clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        conf = confs[i] if len(confs) > i else None
        cls = clss[i] if len(clss) > i else None

        # 获取类别名
        label = model.names[cls] if hasattr(model, "names") and cls is not None else str(cls)
        label_text = f"{label} {conf:.2f}" if conf is not None else label

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 写标签
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# 显示结果窗口
cv2.imshow("YOLO Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图片
cv2.imwrite("detect_result.jpg", img)





















# from ultralytics import YOLO

# model = YOLO("/home/xy/adas_project/yolo/yolo11n.pt")

# results = model("/home/xy/adas_project/yolo/imag.jpg")

# # 取第一个 result
# result = results[0]

# result.show()  # 显示检测结果
# result.save()  # 保存结果