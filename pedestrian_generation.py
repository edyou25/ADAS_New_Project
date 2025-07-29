import carla
import random
import asyncio
from pedestrian_routes import coordinates_list

def create_route_from_coordinates(coordinates):
    return [carla.Location(x=coord[0], y=coord[1], z=coord[2]) for coord in coordinates]

def define_routes(coordinates_list):
    return [create_route_from_coordinates(coords) for coords in coordinates_list]

def spawn_pedestrian_at_location(world, blueprint_library, location, route_idx, ped_idx, spawn_dx=0.5, spawn_dy=0.5):
    pedestrian_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
    dx = random.uniform(-spawn_dx, spawn_dx)
    dy = random.uniform(-spawn_dy, spawn_dy)
    spawn_loc = carla.Location(
        x=location.x + dx,
        y=location.y + dy,
        z=location.z
    )
    transform = carla.Transform(spawn_loc)
    pedestrian = world.try_spawn_actor(pedestrian_bp, transform)
    if pedestrian:
        print(f"[Route {route_idx}] 行人 {ped_idx} 生成成功。")
    else:
        print(f"[Route {route_idx}] 行人 {ped_idx} 生成失败（位置被占用？）")
    return pedestrian

async def move_pedestrian_along_route(
    pedestrian, route, speed, route_idx, ped_idx, stop_event,
    stuck_speed_thresh=0.05, stuck_time_thresh=5.0
):
    waypoint_index = 0
    print(f"[Route {route_idx}] 行人 {ped_idx} 开始移动。")
    stuck_time = 0
    try:
        while waypoint_index < len(route):
            if stop_event.is_set():
                print(f"[Route {route_idx}] 行人 {ped_idx} 收到停止信号，准备销毁。")
                break
            if not pedestrian.is_alive:
                print(f"[Route {route_idx}] 行人 {ped_idx} 已消失。")
                break

            target_location = route[waypoint_index]
            current_location = pedestrian.get_location()
            distance = current_location.distance(target_location)

            # 获取速度并判断卡死
            velocity = pedestrian.get_velocity()
            speed_now = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

            if speed_now < stuck_speed_thresh:
                stuck_time += 0.05
            else:
                stuck_time = 0

            if stuck_time > stuck_time_thresh:
                print(f"[Route {route_idx}] 行人 {ped_idx} 速度低于 {stuck_speed_thresh} m/s 超过 {stuck_time_thresh}s，判定卡死，直接销毁。")
                break

            if distance < 1.0:
                waypoint_index += 1
            else:
                direction = target_location - current_location
                if direction.length() > 0.01:
                    direction = direction.make_unit_vector()
                    random_speed = speed + random.uniform(-0.5, 0.5)
                    pedestrian.apply_control(carla.WalkerControl(
                        direction=carla.Vector3D(direction.x, direction.y, direction.z),
                        speed=random_speed
                    ))

            await asyncio.sleep(0.05)

        print(f"[Route {route_idx}] 行人 {ped_idx} 到达终点或被销毁。")

    except (asyncio.CancelledError, KeyboardInterrupt):
        print(f"[Route {route_idx}] 行人 {ped_idx} task被取消或被中断。")
    except Exception as e:
        print(f"[Route {route_idx}] 行人 {ped_idx} 运行异常: {e}")
    finally:
        try:
            if pedestrian.is_alive:
                pedestrian.destroy()
                print(f"[Route {route_idx}] 行人 {ped_idx} 已销毁。")
        except Exception as e:
            print(f"[Route {route_idx}] 行人 {ped_idx} 销毁异常: {e}")
        try:
            await asyncio.sleep(0.01)
        except Exception:
            pass

async def spawn_and_move_pedestrians(
    world, blueprint_library, route, pedestrians_per_hour, speed, route_idx, initial_batch,
    all_tasks, stop_event, spawn_dx=0.5, spawn_dy=0.5,
    stuck_speed_thresh=0.05, stuck_time_thresh=5.0
):
    for i in range(initial_batch):
        if stop_event.is_set():
            break
        pedestrian = spawn_pedestrian_at_location(world, blueprint_library, route[0], route_idx, i+1, spawn_dx, spawn_dy)
        if pedestrian:
            t = asyncio.create_task(
                move_pedestrian_along_route(
                    pedestrian, route, speed, route_idx, i+1, stop_event,
                    stuck_speed_thresh=stuck_speed_thresh, stuck_time_thresh=stuck_time_thresh
                )
            )
            all_tasks.append(t)
        await asyncio.sleep(0.2)
    ped_idx = initial_batch + 1
    interval = 3600 / pedestrians_per_hour
    while not stop_event.is_set():
        pedestrian = spawn_pedestrian_at_location(world, blueprint_library, route[0], route_idx, ped_idx, spawn_dx, spawn_dy)
        if pedestrian:
            t = asyncio.create_task(
                move_pedestrian_along_route(
                    pedestrian, route, speed, route_idx, ped_idx, stop_event,
                    stuck_speed_thresh=stuck_speed_thresh, stuck_time_thresh=stuck_time_thresh
                )
            )
            all_tasks.append(t)
            ped_idx += 1
            await asyncio.sleep(interval)
        else:
            print(f"[Route {route_idx}] 生成失败，延迟重试。")
            await asyncio.sleep(0.5)

async def main():
    # ======= 所有可调参数集中在这里 =======
    pedestrians_per_hour = 1000      # 每小时每条路线生成行人数量
    pedestrians_speed = 3.5         # 行人目标速度 (m/s)
    initial_batch = 1               # 启动初始批量
    spawn_dx = 0.2                  # 生成x方向扰动范围（米）
    spawn_dy = 0.2                  # 生成y方向扰动范围（米）
    stuck_speed_thresh = 1.0       # 判定卡死的速度阈值(m/s)
    stuck_time_thresh = 3.0         # 低于阈值持续多少秒判定卡死
    carla_host = 'localhost'
    carla_port = 2000
    carla_timeout = 10.0
    # =====================================

    client = carla.Client(carla_host, carla_port)
    client.set_timeout(carla_timeout)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 清理现有行人
    actors = world.get_actors()
    walkers = actors.filter('walker.pedestrian.*')
    for walker in walkers:
        try:
            walker.destroy()
        except Exception as e:
            print(f"销毁actor异常: {e}")
    print(f"已清理现有行人。")

    routes = define_routes(coordinates_list)
    print(f"共加载 {len(routes)} 条路线。")

    all_tasks = []
    stop_event = asyncio.Event()

    tasks = [
        asyncio.create_task(
            spawn_and_move_pedestrians(
                world, blueprint_library, route, pedestrians_per_hour, pedestrians_speed, idx + 1,
                initial_batch, all_tasks, stop_event, spawn_dx=spawn_dx, spawn_dy=spawn_dy,
                stuck_speed_thresh=stuck_speed_thresh, stuck_time_thresh=stuck_time_thresh
            )
        )
        for idx, route in enumerate(routes)
    ]

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print("主任务异常：", e)
    finally:
        stop_event.set()
        await asyncio.sleep(0.5)
        for t in all_tasks:
            t.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        await asyncio.sleep(0.5)
        pending = [t for t in all_tasks if not t.done()]
        if pending:
            print(f"仍有 {len(pending)} 个任务未完成，强制关闭。")
        all_actors = world.get_actors()
        all_walkers = all_actors.filter('walker.pedestrian.*')
        for walker in all_walkers:
            try:
                walker.destroy()
            except Exception as e:
                print(f"销毁actor异常: {e}")
        print('所有行人已销毁，程序已退出。')

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("主进程收到 KeyboardInterrupt，已吞掉。")
    except Exception as e:
        print("主进程异常：", e)