import matplotlib.pyplot as plt
import pedestrian_routes  # 你的 routes 脚本

import re

def extract_rot_labels(filename):
    """解析出rot编号的顺序，返回如[2, 3, 4, ...]"""
    rot_labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r'\s*#\s*rot(\d+)', line)
            if m:
                rot_labels.append(int(m.group(1)))
    return rot_labels

rot_labels = extract_rot_labels('pedestrian_routes.py')
coords_list = pedestrian_routes.coordinates_list

# 检查编号和轨迹数是否匹配
if len(rot_labels) != len(coords_list):
    print(f"警告：rot编号数量({len(rot_labels)})和轨迹数量({len(coords_list)})不一致！")
    # 矫正，按最短长度zip，防止越界
    min_len = min(len(rot_labels), len(coords_list))
    coords_list = coords_list[:min_len]
    rot_labels = rot_labels[:min_len]

plt.figure(figsize=(16, 12))
ax = plt.gca()
colors = plt.cm.get_cmap('tab20', len(coords_list))

all_x, all_y = [], []

for idx, (coords, rot_num) in enumerate(zip(coords_list, rot_labels)):
    if not coords:
        print(f"警告：rot{rot_num} 轨迹点为空，已跳过")
        continue
    x = [p[0] for p in coords]
    y = [p[1] for p in coords]
    all_x.extend(x)
    all_y.extend(y)

    # 绘制轨迹
    ax.plot(x, y, marker='o', label=f'rot{rot_num}', color=colors(idx))
    # 在每一条边的中点上写编号
    for i in range(len(coords) - 1):
        xm = (coords[i][0] + coords[i+1][0]) / 2
        ym = (coords[i][1] + coords[i+1][1]) / 2
        ax.text(xm, ym, f'rot{rot_num}', fontsize=7, color=colors(idx), alpha=0.5, ha='center', va='center', rotation=0)

    # 起点和终点分别标粗体
    ax.text(x[0], y[0], f'rot{rot_num}', fontsize=12, color=colors(idx), weight='bold', ha='right', va='bottom', alpha=1)
    ax.text(x[-1], y[-1], f'rot{rot_num}', fontsize=12, color=colors(idx), weight='bold', ha='left', va='top', alpha=1)

    # 点数过少警告
    if len(coords) < 3:
        print(f"警告：rot{rot_num} 轨迹只有 {len(coords)} 个点，可能数据不完整。")

# 显示所有点坐标以便人工核查（可注释掉）
# for idx, (coords, rot_num) in enumerate(zip(coords_list, rot_labels)):
#     for (xi, yi) in [(p[0], p[1]) for p in coords]:
#         ax.text(xi, yi, f'({xi:.1f},{yi:.1f})', fontsize=5, color='gray', alpha=0.4)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Pedestrian Routes Visualization (Only X/Y, each edge labeled)')
plt.grid(True)
ax.set_aspect('equal')
plt.legend(loc='best', fontsize=10, ncol=2)
if all_x and all_y:
    plt.xlim(min(all_x)-10, max(all_x)+10)
    plt.ylim(min(all_y)-10, max(all_y)+10)
plt.tight_layout()
plt.show()