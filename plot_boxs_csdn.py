import glob
import os
import sys
import cv2
import numpy as np
import math
import random
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def init_camera(world, sensor_type, transform, attached, image_width, image_height, fov):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(image_width))
    camera_bp.set_attribute('image_size_y', str(image_height))
    camera_bp.set_attribute('fov', str(fov))
    camera_rgb = world.spawn_actor(camera_bp, transform, attach_to=attached)
    return camera_rgb

def process_rgb_image(image, image_rgb_queue, queue_len):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    image_rgb_queue.append(array)
    if len(image_rgb_queue) > queue_len:
        image_rgb_queue.pop(0)

def get_camera_intrinsic(width, height, fov):
    focal = 0.5 * width / np.tan(0.5 * np.deg2rad(fov))
    K = np.identity(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    print(f"\n==== 相机内参K ====\n{K}\nfocal: {focal}, width: {width}, height: {height}, fov: {fov}\n==================")
    return K

def get_extrinsic_matrix(camera_transform):
    return np.array(camera_transform.get_matrix())

def world_to_camera_blog(world_points, extrinsic):
    num_pts = world_points.shape[0]
    homo_world = np.concatenate([world_points, np.ones((num_pts,1))], axis=1)  # (N,4)
    extrinsic_inv = np.linalg.inv(extrinsic)  # 世界->相机
    cam_xyz = np.dot(extrinsic_inv, homo_world.T)  # (4,N)
    return cam_xyz

def camera_to_pixel_blog(cam_xyz, K):
    y = cam_xyz[1, :]
    z = cam_xyz[2, :]
    x = cam_xyz[0, :]
    coords = np.vstack([y, -z, x])  # 博客强调的顺序
    points_2d_homo = np.dot(K, coords)
    points_2d = np.zeros((2, coords.shape[1]))
    points_2d[0, :] = points_2d_homo[0, :] / points_2d_homo[2, :]
    points_2d[1, :] = points_2d_homo[1, :] / points_2d_homo[2, :]
    return points_2d

def is_in_camera_fov(camera_tf, actor_loc, camera_fov_deg):
    ego_yaw = np.deg2rad(camera_tf.rotation.yaw)
    forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)])
    ego_xy = np.array([camera_tf.location.x, camera_tf.location.y])
    actor_xy = np.array([actor_loc.x, actor_loc.y])
    to_actor = actor_xy - ego_xy
    to_actor_norm = np.linalg.norm(to_actor)
    if to_actor_norm < 1e-6:
        return False
    to_actor = to_actor / to_actor_norm
    cos_angle = np.dot(forward, to_actor)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
    return angle <= (camera_fov_deg / 2)

def get_bbox_3d(actor):
    bbox = actor.bounding_box
    extent = bbox.extent
    # 8个顶点在局部坐标系
    corners = np.array([
        [ extent.x,  extent.y,  extent.z],
        [ extent.x, -extent.y,  extent.z],
        [-extent.x, -extent.y,  extent.z],
        [-extent.x,  extent.y,  extent.z],
        [ extent.x,  extent.y, -extent.z],
        [ extent.x, -extent.y, -extent.z],
        [-extent.x, -extent.y, -extent.z],
        [-extent.x,  extent.y, -extent.z],
    ])
    # bbox中心和朝向在actor坐标系下
    bbox_transform = carla.Transform(bbox.location, bbox.rotation)
    actor_transform = actor.get_transform()
    world_corners = []
    for corner in corners:
        # 转为carla.Location
        loc = carla.Location(x=corner[0], y=corner[1], z=corner[2])
        # 先转到bbox中心（相对actor），再转到世界
        loc = bbox_transform.transform(loc)
        loc = actor_transform.transform(loc)
        world_corners.append([loc.x, loc.y, loc.z])
    return np.array(world_corners)  # (8, 3)


def draw_3d_box(image, pts, color=(0,255,0), thickness=2):
    """
    在图像上绘制3D立体包围盒
    pts: (8,2)的像素坐标，顺序如下（CARLA默认）：
        0-3: 顶面(逆时针)，4-7: 底面(逆时针)
    """
    pts = pts.astype(int)
    # 顶面
    for i in range(4):
        cv2.line(image, tuple(pts[i]), tuple(pts[(i+1)%4]), color, thickness)
    # 底面
    for i in range(4,8):
        cv2.line(image, tuple(pts[i]), tuple(pts[4 + (i+1)%4]), color, thickness)
    # 侧边
    for i in range(4):
        cv2.line(image, tuple(pts[i]), tuple(pts[i+4]), color, thickness)

def draw_fixed_bbox_on_image(
    image, world, camera_transform, camera_K,
    max_distance=50.0, ego_vehicle=None, camera_fov_deg=90
):
    actors = []
    actors += world.get_actors().filter('vehicle.*')
    actors += world.get_actors().filter('walker.*')
    actors += world.get_actors().filter('bicycle.*')
    actors += world.get_actors().filter('motorcycle.*')
    cam_loc = camera_transform.location
    selected_actors = []

    for actor in actors:
        if ego_vehicle is not None and actor.id == ego_vehicle.id:
            continue
        actor_loc = actor.get_location()
        dist = cam_loc.distance(actor_loc)
        if dist >= max_distance:
            continue
        # 用相机坐标与朝向做FOV筛选
        if not is_in_camera_fov(camera_transform, actor_loc, camera_fov_deg):
            continue
        selected_actors.append((actor, actor_loc))

    extrinsic = get_extrinsic_matrix(camera_transform)

    for actor, center in selected_actors:
        type_id = actor.type_id

        # 统一用真实的bounding_box
        bbox_3d = get_bbox_3d(actor)
        if ('bicycle' in type_id) or ('motorcycle' in type_id):
            box_color = (200, 60, 255)  # 紫色
        elif 'vehicle.' in type_id:
            box_color = (0, 255, 0)     # 绿色
        elif 'walker.pedestrian.' in type_id:
            box_color = (255, 100, 0)   # 橙色
        else:
            box_color = (200, 60, 255) # 紫色

        cam_xyz = world_to_camera_blog(bbox_3d, extrinsic)
        zs = cam_xyz[2, :]
        pixels = camera_to_pixel_blog(cam_xyz, camera_K)
        xs, ys = pixels[0, :], pixels[1, :]
        xs_valid = np.logical_and(xs >= 0, xs < image.shape[1])
        ys_valid = np.logical_and(ys >= 0, ys < image.shape[0])

        # # 宽松筛选
        # if not np.any(zs < 0):
        #     continue
        # if np.sum(xs_valid & ys_valid) < 0:
        #     continue

        pts_2d = np.stack([xs, ys], axis=1)
        draw_3d_box(image, pts_2d, color=box_color, thickness=1)
        # id在顶面左上角
        top_idx = np.argmin(xs[:4] + ys[:4])
        cv2.putText(image, f"id:{actor.id}", (int(xs[top_idx]), int(ys[top_idx])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 1)

    return image, len(selected_actors)



def predict_bicycle_trajectory(vehicle, predict_time=5.0, dt=0.5, wheelbase=2.8):
    # 获取自车状态
    tf = vehicle.get_transform()
    x0, y0, z0 = tf.location.x, tf.location.y, tf.location.z
    yaw = math.radians(tf.rotation.yaw)
    v3d = vehicle.get_velocity()
    v = math.sqrt(v3d.x**2 + v3d.y**2 + v3d.z**2) # m/s
    v = max(v, 3.0)

    # 近似前轮转角
    # 通常CARLA没有直接前轮角度，steer范围[-1,1]，最大角约0.7rad
    steer = vehicle.get_control().steer
    delta_max = 0.35 # 最大转角（弧度），你可根据车型调整
    delta = steer * delta_max

    # 仿真轨迹
    num_steps = int(predict_time / dt)
    traj = []
    x, y, yaw_ = x0, y0, yaw
    for _ in range(num_steps):
        x += v * np.cos(yaw_) * dt
        y += v * np.sin(yaw_) * dt
        yaw_ += v / wheelbase * np.tan(delta) * dt
        traj.append([x, y, z0 + 0.5])  # z略上提，避免地面遮挡
    traj = np.array(traj)  # (N, 3)
    return traj

# def draw_trajectory_on_image(image, traj_world, camera_transform, camera_K, extrinsic, alpha=0.6):
#     cam_xyz = world_to_camera_blog(traj_world, extrinsic)
#     zs = cam_xyz[2, :]
#     valid = zs < 0
#     # if np.sum(valid) < 2:
#     #     return image
#     cam_xyz_valid = cam_xyz[:, valid]
#     pixels = camera_to_pixel_blog(cam_xyz_valid, camera_K)
#     xs, ys = pixels[0, :], pixels[1, :]
#     h, w = image.shape[:2]
#     mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
#     xs, ys = xs[mask], ys[mask]

#     overlay = image.copy()  # 复制一层
#     # 在 overlay 上画粗的青色线
#     for i in range(len(xs)-1):
#         pt1 = (int(xs[i]), int(ys[i]))
#         pt2 = (int(xs[i+1]), int(ys[i+1]))
#         cv2.line(overlay, pt1, pt2, (255, 255, 0), 90)

#     # 融合
#     result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
#     return result


def draw_trajectory_on_image(image, traj_world, camera_transform, camera_K, extrinsic, 
                            alpha=0.5, min_thickness=100, max_thickness=150):
    cam_xyz = world_to_camera_blog(traj_world, extrinsic)
    zs = cam_xyz[2, :]
    valid = zs < 0
    if np.sum(valid) < 2:
        return image
    cam_xyz_valid = cam_xyz[:, valid]
    pixels = camera_to_pixel_blog(cam_xyz_valid, camera_K)
    xs, ys, zvals = pixels[0, :], pixels[1, :], cam_xyz_valid[2, :]
    h, w = image.shape[:2]
    mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    idxs = np.where(mask)[0]  # 保证顺序不乱
    if len(idxs) < 2:
        return image
    xs, ys, zvals = xs[idxs], ys[idxs], zvals[idxs]
    
    overlay = image.copy()
    z_near = -zvals[0]
    z_far = -zvals[-1]
    if abs(z_far - z_near) < 1e-3:
        z_far = z_near + 1e-3  # 防止除零
    for i in range(len(xs)-1):
        z1 = -zvals[i]
        # 线性插值线宽。z_near为近端，z_far为远端
        thickness = int(
            max_thickness - (max_thickness-min_thickness) * ((z1 - z_near)/(z_far - z_near))
        )
        thickness = np.clip(thickness, min_thickness, max_thickness)
        pt1 = (int(xs[i]), int(ys[i]))
        pt2 = (int(xs[i+1]), int(ys[i+1]))
        cv2.line(overlay, pt1, pt2, (255,255,0), thickness)
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result

def draw_steering_arrow(image, steer, mode="lane_change"):
    height, width = image.shape[:2]
    color = (0, 255, 255)
    thickness = 8
    tipLength = 0.3

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 3

    length = int(width * 0.12)

    # 默认：居中下方
    action_text = "Lane Keeping"
    center = (int(width * 0.5), int(height * 0.8))
    end = (center[0], center[1] - length)

    # 文字默认放中下
    textsize = cv2.getTextSize(action_text, font, font_scale, font_thickness)[0]
    text_org = (center[0] - textsize[0] // 2, min(center[1] + 60, height - 10))

    if mode == "lane_change":
        if steer > 0.30:
            # 右转
            action_text = "Right Turn"
            center = (int(width * 0.85), int(height * 0.85))
            end = (center[0] + length, center[1])
        elif steer < -0.30:
            # 左转
            action_text = "Left Turn"
            center = (int(width * 0.15), int(height * 0.85))
            end = (center[0] - length, center[1])
        elif steer > 0.03:
            # 右变道
            action_text = "Right Lane Change"
            center = (int(width * 0.85), int(height * 0.85))
            end = (center[0] + length, center[1])
        elif steer < -0.03:
            # 左变道
            action_text = "Left Lane Change"
            center = (int(width * 0.15), int(height * 0.85))
            end = (center[0] - length, center[1])
        else:
            # 车道保持
            action_text = "Lane Keeping"
            center = (int(width * 0.5), int(height * 0.8))
            end = (center[0], center[1] - length)

        # 统一文字位置
        textsize = cv2.getTextSize(action_text, font, font_scale, font_thickness)[0]
        text_org = (center[0] - textsize[0] // 2, min(center[1] + 60, height - 10))


    # 画箭头
    cv2.arrowedLine(image, center, end, color, thickness, tipLength=tipLength)
    # 画带黑边的文字
    cv2.putText(image, action_text, text_org, font, font_scale, (0, 0, 0), font_thickness+2, cv2.LINE_AA)
    cv2.putText(image, action_text, text_org, font, font_scale, color, font_thickness, cv2.LINE_AA)
    return image


def run_simulation(cfg, client):
    actor_list = []
    try:
        world = client.get_world()
        if cfg['sync_mode']:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.010
            world.apply_settings(settings)

        bp = world.get_blueprint_library().filter('vehicle.*')[0]
        vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)

        IM_WIDTH = cfg['cam_width']
        IM_HEIGHT = cfg['cam_height']
        QUEUE_LEN = cfg['queue_length']
        CAMERA_FOV = cfg['cam_fov']

        # 车载相机
        camera_onboard = init_camera(
            world, 'RGBCamera',
            carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0), carla.Rotation(pitch=0, yaw=0, roll=0)),
            vehicle, IM_WIDTH, IM_HEIGHT, CAMERA_FOV
        )
        actor_list.append(camera_onboard)
        camera_onboard_img_queue = []
        camera_onboard.listen(lambda image: process_rgb_image(image, camera_onboard_img_queue, QUEUE_LEN))

        # road side 相机
        CAMERA_FOV_INTECTION = 120
        camera_intection = init_camera(
            world, 'RGBCamera',
            carla.Transform(carla.Location(x=120.0, y=20.0, z=10), carla.Rotation(pitch=-20, yaw=180, roll=0)),
            None, IM_WIDTH, IM_HEIGHT, CAMERA_FOV_INTECTION
        )
        actor_list.append(camera_intection)
        camera_intection_img_queue = []
        camera_intection.listen(lambda image: process_rgb_image(image, camera_intection_img_queue, QUEUE_LEN))

        K_onboard = get_camera_intrinsic(IM_WIDTH, IM_HEIGHT, CAMERA_FOV)
        K_intection = get_camera_intrinsic(IM_WIDTH, IM_HEIGHT, CAMERA_FOV_INTECTION)
        print(f"camera_onboard.attributes: {camera_onboard.attributes}")
        print(f"camera_intection.attributes: {camera_intection.attributes}")


        target_fps = 30
        target_period = 1.0 / target_fps  # 单位：秒
        while True:
            if cfg['sync_mode']:
                world.tick()
            time_start = time.time()

            ego_tf = vehicle.get_transform()
            ego_velocity_3d = vehicle.get_velocity()
            ego_velocity_ms = math.sqrt(ego_velocity_3d.x**2 + ego_velocity_3d.y**2 + ego_velocity_3d.z**2)
            ego_velocity_kmh = round(ego_velocity_ms * 3.6, 1)
            wheel_angle = 0


            if len(camera_intection_img_queue) > 1:
                image_intection = camera_intection_img_queue[-1].copy()
                camera_intection_tf = camera_intection.get_transform()
                image_intection, num_actors = draw_fixed_bbox_on_image(
                    image_intection, world, camera_intection_tf, K_intection,
                    max_distance=90.0, ego_vehicle=vehicle, camera_fov_deg=CAMERA_FOV_INTECTION)
                cv2.imshow("intection", image_intection)
                
            if len(camera_onboard_img_queue) > 1:
                image_onboard = camera_onboard_img_queue[-1].copy()
                camera_world_tf = camera_onboard.get_transform()

                extrinsic = get_extrinsic_matrix(camera_world_tf)
                # 预测轨迹\画轨迹
                traj = predict_bicycle_trajectory(vehicle, predict_time=3.0, dt=0.1, wheelbase=2.8)
                image_onboard = draw_trajectory_on_image(
                    image_onboard, traj, camera_world_tf, K_onboard, extrinsic
                )
                # 画转向箭头
                steer = vehicle.get_control().steer
                image_onboard = draw_steering_arrow(image_onboard, steer, mode="lane_change")

                # 画3D包围盒
                image_onboard, num_actors = draw_fixed_bbox_on_image(
                    image_onboard, world, camera_world_tf, K_onboard,
                    max_distance=70.0, ego_vehicle=vehicle, camera_fov_deg=CAMERA_FOV)
                print(f"Number of selected actors: {num_actors}")
                speed_inform =       "Speed:       " + str(ego_velocity_kmh) + " km/h"
                wheel_angle_inform = "Wheel angle: " + str(round(wheel_angle,2))
                cv2.putText(image_onboard, speed_inform, (30, 35), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                cv2.putText(image_onboard, wheel_angle_inform, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)

                cv2.imshow("onboard", image_onboard)
                cv2.waitKey(5)

            time_end = time.time()
            elapsed = time_end - time_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            # 可选：打印帧率和本帧耗时
            print(f"Loop time: {elapsed*1000:.2f} ms, FPS: {1.0/max(elapsed, target_period):.2f}")

    finally:
        for actor in actor_list:
            actor.destroy()
        cv2.destroyAllWindows()
        print('destroy all actor')

def main():
    try:
        host = '127.0.0.1'
        port = 2000
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        cfg = {
            'sync_mode': False,
            'cam_width': 1280,
            'cam_height': 720,
            'cam_fov': 90,
            'queue_length': 5,
        }
        run_simulation(cfg, client)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()