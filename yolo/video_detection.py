from ultralytics import YOLO
import cv2
import glob
import os
import numpy as np
from collections import defaultdict

# 生成类别颜色（如类别数大于此列表自动循环用色）
def get_color(idx):
    color_list = [
        (255,56,56),   # 红
        (255,157,151), # 粉
        (255,112,31),  # 橙
        (255,178,29),  # 黄
        (207,210,49),  # 青绿
        (72,249,10),   # 绿
        (146,204,23),  # 草绿
        (61,219,134),  # 青
        (26,147,52),   # 深绿
        (0,212,187),   # 蓝绿
        (44,153,168),  # 蓝
        (0,194,255),   # 高亮蓝
        (52,69,147),   # 深蓝
        (100,115,255), # 紫
        (0,24,236),    # 靛蓝
        (132,56,255),  # 紫红
        (82,0,133),    # 深紫
        (203,56,255),  # 紫粉
        (255,149,200), # 浅粉
        (255,55,199),  # 粉紫
    ]
    return color_list[idx % len(color_list)]

# 青色轨迹线
TRACK_COLOR = (255, 255, 0)  # BGR: 青色

# 加载模型
model = YOLO("/home/xy/adas_project/yolo/yolo11n.pt")

video_list = glob.glob("*.mp4") + glob.glob("*.avi") + glob.glob("*.mov") + glob.glob("*.mkv")

for video_path in video_list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        continue

    save_path = os.path.splitext(video_path)[0] + "_track.mp4"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print(f"正在跟踪并可视化: {video_path}")

    # 每个目标的轨迹历史点
    track_history = defaultdict(list)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=0.3,
            device="cuda"
        )

        result = results[0]
        boxes = result.boxes
        if boxes is not None and hasattr(boxes, "id") and boxes.id is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
            clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []
            ids = boxes.id.cpu().numpy().astype(int) if hasattr(boxes, "id") else [None]*len(xyxy)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                conf = confs[i] if len(confs) > i else None
                cls = clss[i] if len(clss) > i else None
                track_id = ids[i] if ids is not None and len(ids) > i else None

                # 不同类别不同颜色
                color = get_color(cls)
                label = model.names[cls] if hasattr(model, "names") and cls is not None else str(cls)
                label_text = f"ID{track_id} {label} {conf:.2f}" if conf is not None else f"ID{track_id} {label}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 轨迹点（中心点）
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                if track_id is not None:
                    track = track_history[track_id]
                    track.append((x_center, y_center))
                    if len(track) > 30:
                        track.pop(0)
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    # 轨迹线用青色
                    cv2.polylines(frame, [points], isClosed=False, color=TRACK_COLOR, thickness=3)

        cv2.imshow(f"Tracking {video_path}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"跟踪完成，结果已保存为: {save_path}")




# from ultralytics import YOLO
# import cv2
# import glob
# import os

# # 加载模型
# model = YOLO("/home/xy/adas_project/yolo/yolo11n.pt")

# # 获取当前目录下所有视频文件（可根据实际后缀调整）
# video_list = glob.glob("*.mp4") + glob.glob("*.avi") + glob.glob("*.mov") + glob.glob("*.mkv")

# for video_path in video_list:
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"无法打开视频: {video_path}")
#         continue

#     # 输出文件名
#     save_path = os.path.splitext(video_path)[0] + "_detect.mp4"

#     # 视频参数
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

#     print(f"正在处理: {video_path}")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 推理检测（可设置device="cuda"加速）
#         results = model(frame)
#         result = results[0]

#         # 获取检测框
#         boxes = result.boxes
#         if boxes is not None and len(boxes) > 0:
#             xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
#             confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
#             clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []

#             for i in range(len(xyxy)):
#                 x1, y1, x2, y2 = map(int, xyxy[i])
#                 conf = confs[i] if len(confs) > i else None
#                 cls = clss[i] if len(clss) > i else None

#                 # 类别名
#                 label = model.names[cls] if hasattr(model, "names") and cls is not None else str(cls)
#                 label_text = f"{label} {conf:.2f}" if conf is not None else label

#                 # 画框和标签
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

#         # 显示实时检测帧（可选）
#         cv2.imshow(f"Detecting {video_path}", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         # 写入到新视频
#         writer.write(frame)

#     cap.release()
#     writer.release()
#     cv2.destroyAllWindows()
#     print(f"检测完成，结果已保存为: {save_path}")