import math
import carla
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np


def get_prediction(cur_tf, vel, steer, horizon, dt):
    trajectory = []
    L = 4.0
    x0, y0, yaw0 = cur_tf.location.x, cur_tf.location.y, math.radians(cur_tf.rotation.yaw)
    # trajectory.append([x0, y0, yaw0, vel])
    trajectory.append([x0, y0, yaw0, vel])
    for idx in range(horizon):
        x1 = x0 + vel*math.cos(yaw0)*dt
        y1 = y0 + vel*math.sin(yaw0)*dt
        yaw1 = yaw0 + vel / L * math.tan(steer)*dt
        x0, y0, yaw0 = x1, y1, yaw1
        trajectory.append([x1, y1, yaw1, vel])

    return trajectory



horizon, dt = 8, 0.2
ego_speed = 10
ego_steer = 0.03
ego_tf   = carla.Transform(carla.Location(x=-160, y=100, z=5), 
                           carla.Rotation(pitch=0, yaw=35, roll=0))
other_speed = 10
other_steer = 0.03
other_tf = carla.Transform(carla.Location(x=-250, y=95, z=5), 
                           carla.Rotation(pitch=0, yaw=0, roll=0))
ego_trajectory = get_prediction(ego_tf, ego_speed, ego_steer, horizon, dt)
other_trajectory = get_prediction(other_tf, other_speed, other_steer, horizon, dt)

plt.figure(figsize=(18,6), dpi=90)
p1 = plt.subplot(1, 3, 1)
ego_x = [row[0] for row in ego_trajectory]
ego_y = [row[1] for row in ego_trajectory]
p1.plot(ego_x, ego_y, label='ego')
p1.scatter(ego_x[0], ego_y[0])
other_x = [row[0] for row in other_trajectory]
other_y = [row[1] for row in other_trajectory]
p1.plot(other_x, other_y, label='other')
p1.scatter(other_x[0], other_y[0])
plt.axis('equal')
plt.legend()


delta_x = other_tf.location.x - ego_tf.location.x
delta_y = other_tf.location.y - ego_tf.location.y
delta_yaw = math.radians(other_tf.rotation.yaw - ego_tf.rotation.yaw)
distance_scale = 5
speed_scale = 5
steer_scale = 1/speed_scale
diy_ego_tf   = carla.Transform(carla.Location(x=0, y=0, z=0), 
                               carla.Rotation(pitch=0, yaw= 90, roll=0))
# 旋转至 ego frame
diy_other_tf = carla.Transform(
               carla.Location(x=(delta_x*math.cos(delta_yaw) - delta_y*math.sin(delta_yaw))*distance_scale,
                              y=(delta_x*math.sin(delta_yaw) + delta_y*math.cos(delta_yaw))*distance_scale, z=0), 
               carla.Rotation(pitch=0, yaw=other_tf.rotation.yaw - ego_tf.rotation.yaw + 0, roll=0))
# 再转 90°
diy_other_tf = carla.Transform(
                carla.Location(x = -diy_other_tf.location.y, y = diy_other_tf.location.x, z=0), 
                carla.Rotation(pitch=0, yaw=diy_other_tf.rotation.yaw + 90, roll=0))
diy_ego_traj   = get_prediction(diy_ego_tf, ego_speed*speed_scale, ego_steer*steer_scale, horizon, dt)
diy_other_traj = get_prediction(diy_other_tf, other_speed*speed_scale, other_steer*steer_scale, horizon, dt)
p2 = plt.subplot(1, 3, 2)
diy_ego_x = [row[0] for row in diy_ego_traj]
diy_ego_y = [row[1] for row in diy_ego_traj]
p2.plot(diy_ego_x, diy_ego_y, label='diy_ego')
p2.scatter(diy_ego_x[0], diy_ego_y[0])
diy_other_x = [row[0] for row in diy_other_traj]
diy_other_y = [row[1] for row in diy_other_traj]
p2.plot(diy_other_x, diy_other_y, label='diy_other')
p2.scatter(diy_other_x[0], diy_other_y[0])
plt.axis('equal')
plt.legend()


x_offset = 100
y_offset = 100
p3 = plt.subplot(1, 3, 3)
image = Image.open("test.png")
image_array = np.array(image)
p3.imshow(image_array)
# y 轴反向 + 平移
diy_ego_x = [row[0]+x_offset  for row in diy_ego_traj]
diy_ego_y = [-row[1]+y_offset for row in diy_ego_traj]
p3.plot(diy_ego_x, diy_ego_y, label='diy_ego')
p3.scatter(diy_ego_x[0], diy_ego_y[0])
diy_other_x = [row[0]+x_offset  for row in diy_other_traj]
diy_other_y = [-row[1]+y_offset for row in diy_other_traj]
p3.plot(diy_other_x, diy_other_y, label='diy_other')
p3.scatter(diy_other_x[0], diy_other_y[0])
plt.axis('equal')
plt.legend()

plt.tight_layout()
plt.show()
