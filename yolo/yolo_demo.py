from ultralytics import YOLO

model = YOLO("/home/xy/adas_project/yolo/yolo11n.pt")

results = model("/home/xy/adas_project/yolo/imag.jpg")

# 取第一个 result
result = results[0]

result.show()  # 显示检测结果
result.save()  # 保存结果