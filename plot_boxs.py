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
    # focal的单位是像素，fov为水平视场角
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    # focal = 800   # 可手动试800/1000验证链路
    K = np.array([
        [focal, 0, width / 2.0],
        [0, focal, height / 2.0],
        [0, 0, 1.0]
    ])
    print(f"\n==== 相机内参K ====\n{K}\nfocal: {focal}, width: {width}, height: {height}, fov: {fov}\n==================")
    return K

def world_to_camera(world_points, camera_transform):
    """
    world_points: (N,3)
    camera_transform: carla.Transform
    输出: (N,3) 在相机坐标系下的点
    """
    # 直接用逆矩阵，无需reshape和.T
    T_world2cam = np.array(camera_transform.get_inverse_matrix())
    num_pts = world_points.shape[0]
    homo_world = np.concatenate([world_points, np.ones((num_pts,1))], axis=1)  # (N,4)
    camera_points_homo = np.dot(homo_world, T_world2cam.T)  # (N,4)
    camera_points = camera_points_homo[:,:3]
    camera_points[:,1] *= -1  # Y轴取反，保持和OpenCV一致
    camera_points[:,2] *= -1  # 新加，反转z轴
    return camera_points

def camera_to_pixel(camera_points, K):
    z = camera_points[:, 2:3]
    z[z==0] = 1e-6
    points_2d = np.dot(K, camera_points.T).T
    points_2d = points_2d[:, :2] / z
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

    print("==== 选中actor的tf信息 ====")
    for actor, loc in selected_actors:
        tf = actor.get_transform()
        print(f"actor id={actor.id}, type={actor.type_id}, "
              f"location=({tf.location.x:.2f},{tf.location.y:.2f},{tf.location.z:.2f}), "
              f"rotation=({tf.rotation.pitch:.2f},{tf.rotation.yaw:.2f},{tf.rotation.roll:.2f})")
    print("=========================")

    for actor, center in selected_actors:
        world_center = np.array([[center.x, center.y, center.z]])
        camera_center = world_to_camera(world_center, camera_transform)  # shape (1,3)
        z = camera_center[0,2]

        # Debug信息打印
        print(f"actor id={actor.id}, world=({center.x:.2f},{center.y:.2f},{center.z:.2f}), "
              f"cam_xyz=({camera_center[0,0]:.2f},{camera_center[0,1]:.2f},{camera_center[0,2]:.2f}), "
              f"x/z={camera_center[0,0]/(z+1e-8):.2f}, y/z={camera_center[0,1]/(z+1e-8):.2f}")

        if z <= 0:
            print(f"actor id={actor.id}: 跳过，因为z={z:.2f}，目标在相机后方")
            continue  # 只画相机前方目标

        image_point = camera_to_pixel(camera_center, camera_K)[0]
        x, y = int(image_point[0]), int(image_point[1])
        print(f"actor id={actor.id}, pixel=({x},{y}), image shape=({image.shape[1]},{image.shape[0]})")

        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 52, (0, 255, 0), 30)
            cv2.putText(image, f"id:{actor.id}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            print(f"actor id={actor.id}: 跳过，因为像素({x},{y})不在画幅内")

    return image, len(selected_actors)

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
            carla.Transform(carla.Location(x=2.5, z=1.7), carla.Rotation(pitch=0, yaw=0)),
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
                    max_distance=50.0, ego_vehicle=vehicle, camera_fov_deg=CAMERA_FOV)
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