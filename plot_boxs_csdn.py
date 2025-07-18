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
    """
    得到相机的4x4外参矩阵，相机->世界
    """
    return np.array(camera_transform.get_matrix())

def world_to_camera_blog(world_points, extrinsic):
    """
    world_points: (N,3)
    extrinsic: 4x4 相机->世界
    返回 shape (4, N)
    """
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

def is_in_camera_fov(ego_tf, actor_loc, camera_fov_deg):
    ego_yaw = np.deg2rad(ego_tf.rotation.yaw)
    forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)])
    ego_xy = np.array([ego_tf.location.x, ego_tf.location.y])
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
    """
    获取actor的3D bounding box八个顶点的世界坐标
    """
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
    max_distance=50.0, ego_vehicle=None, camera_fov_deg=70
):
    actors = []
    actors += world.get_actors().filter('vehicle.*')
    actors += world.get_actors().filter('walker.*')
    cam_loc = camera_transform.location
    selected_actors = []

    if ego_vehicle is not None:
        ego_tf = ego_vehicle.get_transform()
    else:
        ego_tf = None

    for actor in actors:
        if ego_vehicle is not None and actor.id == ego_vehicle.id:
            continue
        actor_loc = actor.get_location()
        dist = cam_loc.distance(actor_loc)
        if dist >= max_distance:
            continue
        if ego_tf is not None and not is_in_camera_fov(ego_tf, actor_loc, camera_fov_deg):
            continue
        selected_actors.append((actor, actor_loc))

    extrinsic = get_extrinsic_matrix(camera_transform)

    for actor, center in selected_actors:
        type_id = actor.type_id

        # 判断是否是自行车或摩托车
        if ('vehicle.bicycle' in type_id) or ('vehicle.motorcycle' in type_id):
            # 预设尺寸
            preset_extent = carla.Vector3D(0.9, 0.3, 0.8)   # 长宽高的一半
            bbox_location = actor.bounding_box.location
            bbox_rotation = actor.bounding_box.rotation
            bbox_transform = carla.Transform(bbox_location, bbox_rotation)
            actor_transform = actor.get_transform()
            # 计算8个顶点
            corners = np.array([
                [ preset_extent.x,  preset_extent.y,  preset_extent.z],
                [ preset_extent.x, -preset_extent.y,  preset_extent.z],
                [-preset_extent.x, -preset_extent.y,  preset_extent.z],
                [-preset_extent.x,  preset_extent.y,  preset_extent.z],
                [ preset_extent.x,  preset_extent.y, -preset_extent.z],
                [ preset_extent.x, -preset_extent.y, -preset_extent.z],
                [-preset_extent.x, -preset_extent.y, -preset_extent.z],
                [-preset_extent.x,  preset_extent.y, -preset_extent.z],
            ])
            world_corners = []
            for corner in corners:
                loc = carla.Location(x=corner[0], y=corner[1], z=corner[2])
                loc = bbox_transform.transform(loc)
                loc = actor_transform.transform(loc)
                world_corners.append([loc.x, loc.y, loc.z])
            bbox_3d = np.array(world_corners)
            box_color = (200, 60, 255)  # 紫色
        else:
            bbox_3d = get_bbox_3d(actor)
            # 其它类型颜色
            if 'vehicle.' in type_id:
                box_color = (0, 255, 0)
            elif 'walker.pedestrian.' in type_id:
                box_color = (255, 100, 0)
            else:
                box_color = (128, 128, 128)

        cam_xyz = world_to_camera_blog(bbox_3d, extrinsic)
        zs = cam_xyz[2, :]
        pixels = camera_to_pixel_blog(cam_xyz, camera_K)
        xs, ys = pixels[0, :], pixels[1, :]
        xs_valid = np.logical_and(xs >= 0, xs < image.shape[1])
        ys_valid = np.logical_and(ys >= 0, ys < image.shape[0])

        # 宽松筛选
        if not np.any(zs < 0):
            continue
        if np.sum(xs_valid & ys_valid) < 2:
            continue

        pts_2d = np.stack([xs, ys], axis=1)
        draw_3d_box(image, pts_2d, color=box_color, thickness=1)
        # id在顶面左上角
        top_idx = np.argmin(xs[:4] + ys[:4])
        cv2.putText(image, f"id:{actor.id}", (int(xs[top_idx]), int(ys[top_idx])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 1)

    return image, len(selected_actors)

# def draw_fixed_bbox_on_image(
#     image, world, camera_transform, camera_K,
#     max_distance=100.0, ego_vehicle=None, camera_fov_deg=70
# ):
#     actors = []
#     actors += world.get_actors().filter('vehicle.*')
#     actors += world.get_actors().filter('walker.*')
#     cam_loc = camera_transform.location
#     selected_actors = []

#     if ego_vehicle is not None:
#         ego_tf = ego_vehicle.get_transform()
#     else:
#         ego_tf = None

#     for actor in actors:
#         if ego_vehicle is not None and actor.id == ego_vehicle.id:
#             continue
#         actor_loc = actor.get_location()
#         dist = cam_loc.distance(actor_loc)
#         if dist >= max_distance:
#             continue
#         if ego_tf is not None and not is_in_camera_fov(ego_tf, actor_loc, camera_fov_deg):
#             continue
#         selected_actors.append((actor, actor_loc))

#     extrinsic = get_extrinsic_matrix(camera_transform)

#     for actor, center in selected_actors:
#         bbox_3d = get_bbox_3d(actor)  # (8,3)
#         cam_xyz = world_to_camera_blog(bbox_3d, extrinsic)  # (4,8)
#         zs = cam_xyz[2, :]
#         pixels = camera_to_pixel_blog(cam_xyz, camera_K)  # (2,8)
#         xs, ys = pixels[0, :], pixels[1, :]
#         xs_valid = np.logical_and(xs >= 0, xs < image.shape[1])
#         ys_valid = np.logical_and(ys >= 0, ys < image.shape[0])

#         # 只要有至少1个点在前方（z<0），就绘制
#         if not np.any(zs < 0):
#             continue
#         # 只要有2个点在画幅内，就绘制
#         if np.sum(xs_valid & ys_valid) < 2:
#             continue

#         pts_2d = np.stack([xs, ys], axis=1)  # (8,2)

#         # 按类型分颜色
#         if 'vehicle.' in actor.type_id:
#             color = (0, 255, 0)    # 绿色
#         elif 'walker.pedestrian.' in actor.type_id:
#             color = (255, 100, 0)  # 橙色
#         else:
#             color = (128, 128, 128)

#         draw_3d_box(image, pts_2d, color=color, thickness=1)
#         # id在顶面左上角
#         top_idx = np.argmin(xs[:4] + ys[:4])
#         cv2.putText(image, f"id:{actor.id}", (int(xs[top_idx]), int(ys[top_idx])-5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

#     return image, len(selected_actors)


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

        # 相机位置推荐放在车头前方2.5米、离地1.7米、俯视10度
        camera_rgb = init_camera(
            world, 'RGBCamera',
            carla.Transform(carla.Location(x=0.0, z=2.4), carla.Rotation(pitch=0, yaw=0)),
            vehicle, IM_WIDTH, IM_HEIGHT, CAMERA_FOV
        )
        actor_list.append(camera_rgb)
        image_rgb_queue = []
        camera_rgb.listen(lambda image: process_rgb_image(image, image_rgb_queue, QUEUE_LEN))

        K = get_camera_intrinsic(IM_WIDTH, IM_HEIGHT, CAMERA_FOV)
        print(f"camera_rgb.attributes: {camera_rgb.attributes}")

        while True:
            if cfg['sync_mode']:
                world.tick()
            time_start = time.time()

            ego_tf = vehicle.get_transform()
            ego_velocity_3d = vehicle.get_velocity()
            ego_velocity_ms = math.sqrt(ego_velocity_3d.x**2 + ego_velocity_3d.y**2 + ego_velocity_3d.z**2)
            ego_velocity_kmh = round(ego_velocity_ms * 3.6, 1)
            wheel_angle = 0

            spectator = world.get_spectator()
            ego_yaw = math.radians(vehicle.get_transform().rotation.yaw)
            spectator.set_transform(carla.Transform(
                ego_tf.location + carla.Location(x=-10*math.cos(ego_yaw), y=-10*math.sin(ego_yaw), z=7.5),
                carla.Rotation(pitch=-30, yaw=math.degrees(ego_yaw)))
            )

            if len(image_rgb_queue) > 1:
                image_rgb = image_rgb_queue[-1].copy()
                camera_world_tf = camera_rgb.get_transform()
                image_rgb, num_actors = draw_fixed_bbox_on_image(
                    image_rgb, world, camera_world_tf, K,
                    max_distance=70.0, ego_vehicle=vehicle, camera_fov_deg=CAMERA_FOV)
                print(f"Number of selected actors: {num_actors}")

                # 打印自车tf
                ego_tf = vehicle.get_transform()
                print(f"==== 自车 ego tf ====")
                print(f"ego id={vehicle.id}, "
                      f"location=({ego_tf.location.x:.2f},{ego_tf.location.y:.2f},{ego_tf.location.z:.2f}), "
                      f"rotation=({ego_tf.rotation.pitch:.2f},{ego_tf.rotation.yaw:.2f},{ego_tf.rotation.roll:.2f})")
                print("=====================")

                # 打印相机tf
                camera_tf = camera_rgb.get_transform()
                print(f"==== 相机 camera tf ====")
                print(f"camera id={camera_rgb.id}, "
                      f"location=({camera_tf.location.x:.2f},{camera_tf.location.y:.2f},{camera_tf.location.z:.2f}), "
                      f"rotation=({camera_tf.rotation.pitch:.2f},{camera_tf.rotation.yaw:.2f},{camera_tf.rotation.roll:.2f})")
                print("======================")

                speed_inform =       "Speed:       " + str(ego_velocity_kmh) + " km/h"
                wheel_angle_inform = "Wheel angle: " + str(round(wheel_angle,2))
                cv2.putText(image_rgb, speed_inform, (30, 35), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                cv2.putText(image_rgb, wheel_angle_inform, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                cv2.imshow("output", image_rgb)
                cv2.waitKey(5)

            time_end = time.time()

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