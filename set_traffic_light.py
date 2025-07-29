import carla

def set_all_traffic_lights_to_green(world):
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')
    count = 0
    for light in traffic_lights:
        light.set_state(carla.TrafficLightState.Green)
        light.freeze(True)  # 冻结为绿灯状态
        count += 1
    print(f"设置了 {count} 个红绿灯为绿灯。")

if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    set_all_traffic_lights_to_green(world)
    print("所有红绿灯已设置为绿灯，并已冻结。")