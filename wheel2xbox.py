import pygame
import math
import vgamepad as vg
import logging
import time
from logging.handlers import RotatingFileHandler

# 设置日志
# log_handler = RotatingFileHandler('wheel_controller.log', maxBytes=1024*1024, backupCount=5)
# logging.basicConfig(handlers=[log_handler], level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

def map_steering_to_trigger(steering_value, deadzone=0.02):
    if abs(steering_value) < deadzone:
        return 0.5
    mapped_value = math.copysign(steering_value**2, steering_value)
    return (mapped_value + 1) / 2

def run_program():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        logging.error("没有检测到方向盘")
        return

    wheel = pygame.joystick.Joystick(0)
    wheel.init()

    gamepad = vg.VX360Gamepad()

    running = True
    reversing = False
    REVERSE = 5
    first = True

    print('方向盘初始化中....')
    while first:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                first = False
        throttle = wheel.get_axis(1)
        if throttle >= 0.98:
            first = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.JOYBUTTONDOWN:
                if REVERSE == event.button:
                    reversing = not reversing

        steering = wheel.get_axis(0)
        throttle = -1 * (wheel.get_axis(1) - 1) /2
        braking = wheel.get_axis(2)
        is_brake = braking < 0.5

        # trigger_value = map_steering_to_trigger(steering)

        trans_steering = -1 * (steering - 1) /2
        gamepad.left_trigger_float(value_float=trans_steering)
        gamepad.right_trigger_float(value_float=throttle)

        if is_brake:
            gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        else:
            gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        
        if reversing:
            gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        else:
            gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)

        gamepad.update()

        print(f"Steering: {steering:.2f}, throttle: {throttle:.2f}, brake: {is_brake}, reverse: {reversing}")

    pygame.quit()

if __name__ == "__main__":
    run_program()