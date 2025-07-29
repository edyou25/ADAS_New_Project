"""
Description: Run the script to get real-time prediction
"""
import functools
import carla
import random
import asyncio
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import torch
import cv2
from multiprocessing import Process, Queue

from routes import coordinates_list
from porcess_trajectory import Processor
from predict_trajectory import Predictor

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import *
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


def catch_exceptions(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            print(f"An exception occurred in {func.__name__}: {e}")

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An exception occurred in {func.__name__}: {e}")

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def get_pedestrian_locations(world):
    actors = world.get_actors()
    pedestrians = actors.filter('walker.pedestrian.*')
    locations = [(pedestrian.id, pedestrian.get_location()) for pedestrian in pedestrians]
    return locations

def draw_trajectory(ax, pred_trajs, topk_scores):

    if isinstance(topk_scores, torch.Tensor):
        topk_scores = topk_scores.cpu().numpy().flatten()

    # 绘制单个轨迹
    for traj, score in zip(pred_trajs, topk_scores):
        # 确保 traj 是 CPU 上的 NumPy 数组
        if isinstance(traj, torch.Tensor):
            traj = traj.cpu().numpy()

        # 只取前五个点
        traj = traj[:5]
        #traj_scaled = (traj - np.array([min_x, min_y])) * scale_factor
        traj_scaled = traj
        ax.plot(traj_scaled[:, 1], traj_scaled[:, 0])
        ax.scatter(traj_scaled[:, 1], traj_scaled[:, 0])

        # 在轨迹终点标注分数
        end_point = traj_scaled[-1]
        ax.annotate(f'{score:.2f}', (end_point[1], end_point[0]), textcoords="offset points", xytext=(0,10), ha='center')
        
    return ax

def draw_all_predictions(predictions, screen, boundary):
    x_min, x_max, y_min, y_max = boundary

    width, height = screen.get_size()  # 获取 Pygame 窗口尺寸
    screen.fill((0, 0, 0))

    # 创建一个空白图像
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    # ax.set_xlim(x_min-20, x_max+20)
    # ax.set_ylim(y_min-20, y_max+20)
    ax.set_xlim(0,-200)
    ax.set_ylim(50,150)
    ax.grid()

    # # 反转y轴
    plt.gca().invert_xaxis()
    
    # 绘制所有预测轨迹
    for id, (pred_trajs, topk_scores) in predictions.items():
        if pred_trajs is not None:
    
            ax = draw_trajectory(ax, pred_trajs, topk_scores)

    # 保存图像到临时文件
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 将图像从 RGB 转换为 BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_surface = pygame.surfarray.make_surface(image_bgr.swapaxes(0, 1))
    
    # 绘制图像到屏幕
    screen.blit(image_surface, (0, 0))
    pygame.display.flip()
    plt.close(fig)

def pygame_thread(draw_queue):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    running = True
    while running:
        while not draw_queue.empty():
            predictions, boundary = draw_queue.get()
            draw_all_predictions(predictions, screen, boundary)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()
    pygame.quit()

def predict_trajectories(Pre, Tra, id, ped, neis, neis_mask):
    pred_trajs, top_k_scores = Pre.predict(ped, neis, neis_mask)
    pred_trajs = Tra.reverse_trajectory(id,pred_trajs)

    # 将 top_k_scores 转换为列表
    top_k_scores = top_k_scores.cpu().tolist() 
    top_k_scores = top_k_scores[0]

    return pred_trajs, top_k_scores

def concurrent_predictions(Predictor, Trajector, data):
    # data should be a dictionary {id: (ped, neis, neis_mask)}
    predictions = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(predict_trajectories, Predictor, Trajector, id, *data[id]): id for id in data}
        for future in as_completed(futures):
            id = futures[future]
            try:
                pred_trajs, topk_scores = future.result()
                predictions[id] = (pred_trajs, topk_scores)

            except Exception as exc:
                print(f'{id} generated an exception: {exc}')

    return predictions

# N x L x 2
@catch_exceptions
def construct_fixed_length_trajectories(trajectory_data):
    #print("Starting to construct fixed length trajectories...")
    all_ped_ids = set(pid for frame in trajectory_data.values() for pid in frame)
    fixed_length_data = {pid: [] for pid in all_ped_ids}

    for frame_index, frame_data in trajectory_data.items():
        #print(f"Processing frame {frame_index}: {frame_data}")
        for pid in all_ped_ids:
            if pid in frame_data:
                fixed_length_data[pid].append(frame_data[pid])
            else:
                #fixed_length_data[pid].append([np.nan, np.nan, np.nan])
                fixed_length_data[pid].append([np.nan, np.nan])

    return fixed_length_data

@catch_exceptions
async def log_pedestrian_locations(world, boundary, draw_queue, interval=0.8, obs_length=8):

    trajectory_data = {}
    frame_index = 0

    Tra = Processor(dist_threshold=2)
    Pre = Predictor()

    x_min, x_max, y_min, y_max = boundary
    
    while True:
        locations = get_pedestrian_locations(world)
        frame_data = {}
        
        for ped_id, location in locations:
            if x_min <= location.x <= x_max and y_min <= location.y <= y_max:
                #frame_data[ped_id] = [location.x, location.y, location.z]
                frame_data[ped_id] = [location.x, location.y]
        
        trajectory_data[frame_index] = frame_data
        frame_index += 1
        
        if len(trajectory_data) > obs_length:
            oldest_frame = min(trajectory_data.keys())
            del trajectory_data[oldest_frame]

        if len(trajectory_data) == obs_length:
            processed_data = construct_fixed_length_trajectories(trajectory_data)
            Tra.trajectories = processed_data
            data = Tra.transform_trajectory()
            predictions = concurrent_predictions(Pre, Tra, data)

            print(len(predictions))

            # 直接从字典中提取前两个项
            # first_two = dict(list(predictions.items())[:2])

            draw_queue.put((predictions, boundary))

        await asyncio.sleep(interval)

@catch_exceptions
def spawn_pedestrian_at_location(world, blueprint_library, location):
    pedestrian_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
    transform = carla.Transform(location)
    
    pedestrian = world.spawn_actor(pedestrian_bp, transform)
    #print("Spawn success: ", location)
    return pedestrian
    
async def move_pedestrian_along_route(pedestrian, route, speed=5.0):
    waypoint_index = 0
    while waypoint_index < len(route):
        target_location = route[waypoint_index]
        current_location = pedestrian.get_location()
        distance = current_location.distance(target_location)
        
        if distance < 1.0:  # If the pedestrian is close to the target location, move to the next waypoint
            waypoint_index += 1
        else:
            direction = target_location - current_location
            direction = direction.make_unit_vector()
            # Apply random noise to the speed
            random_speed = speed + random.uniform(-0.5, 0.5)
            pedestrian.apply_control(carla.WalkerControl(direction=carla.Vector3D(direction.x, direction.y, direction.z), speed=random_speed))
        
        await asyncio.sleep(0.05)  # Small delay to simulate real-time movement
    
    # Destroy the pedestrian after reaching the final destination
    pedestrian.destroy()

def create_route_from_coordinates(coordinates):
    return [carla.Location(x=coord[0], y=coord[1], z=coord[2]) for coord in coordinates]

def define_routes(coordinates_list):
    # Convert coordinates to routes
    routes = [create_route_from_coordinates(coords) for coords in coordinates_list]
    return routes

async def spawn_and_move_pedestrians(world, blueprint_library, route, pedestrians_per_hour, speed):
    interval = 7200 / pedestrians_per_hour
    #interval = 7200 / pedestrians_per_hour
    while True:
        pedestrian = spawn_pedestrian_at_location(world, blueprint_library, route[0])
        if pedestrian:
            asyncio.create_task(move_pedestrian_along_route(pedestrian, route, speed))
            await asyncio.sleep(interval)

def get_xxyy(x, y, z, pitch):
    # 视场角，一般情况下可以设置为60度至90度
    fov = math.radians(90)  # 转换为弧度
    
    if pitch == -90:
        horizontal_extent = 2 * z * math.tan(fov / 2)
    else:
        # 对于非垂直向下的情况，计算视线与地面的交点距离
        # pitch 需要转换为弧度并取绝对值
        pitch_rad = math.radians(abs(pitch))
        horizontal_extent = z * math.tan(math.pi/2 - pitch_rad + fov / 2)

    # 计算四个边界
    x_min = x - horizontal_extent / 2
    x_max = x + horizontal_extent / 2
    y_min = y - horizontal_extent / 2
    y_max = y + horizontal_extent / 2

    return x_min, x_max, y_min, y_max

def set_spectator_view(client, x, y, z, pitch, yaw, roll):
    spectator = client.get_world().get_spectator()
    transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
    spectator.set_transform(transform)

async def main(pedestrians_per_hour, speed, draw_queue):
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    # Remove all existing pedestrians
    actors = world.get_actors()
    walkers = actors.filter('walker.pedestrian.*')
    for walker in walkers:
        walker.destroy()
    
    # Get routes
    routes = define_routes(coordinates_list)

    # 设置观察者视角
    set_spectator_view(client, x=0, y=-60, z=80, pitch=-90, yaw=0, roll=0)
    #boundary = get_xxyy(80, -30, 80, -90)
    boundary = get_xxyy(0, -60, 80, -90)
    
    try:
        # # 创建绘图线程
        # thread = threading.Thread(target=pygame_thread, args=(draw_queue,))
        # thread.start()
        # Spawn and move pedestrians for each route
        tasks = [spawn_and_move_pedestrians(world, blueprint_library, route, pedestrians_per_hour, speed) for route in routes]
        # Collect pedestrian trajectories and visualize them 
        tasks.append(log_pedestrian_locations(world, boundary, draw_queue))
        
        await asyncio.gather(*tasks, return_exceptions=True)

        
    except Exception as e:
        print(f"An error occurred: {e}")  

    finally:
        # # Ensure all walkers are destroyed
        # actors = world.get_actors()
        # walkers = actors.filter('walker.pedestrian.*')
        # for walker in walkers:
        #     walker.destroy()
        # Properly quit pygame
        pass


if __name__ == '__main__':
    draw_queue = Queue()
    p = Process(target=pygame_thread, args=(draw_queue,))
    p.start()

    pedestrians_per_hour = 500  # Set the number of pedestrians per hour here
    pedestrians_speed = 2 # speed

    asyncio.run(main(pedestrians_per_hour, pedestrians_speed, draw_queue))
