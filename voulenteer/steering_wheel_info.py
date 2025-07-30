import pygame
import sys
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("没有找到方向盘，请检查连接。")
    sys.exit()

# 获取第一个方向盘
joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"连接的方向盘: {joystick.get_name()}")

STEER = 0
THROTTLE = 1
BRAKE = 2
REVERSE = 5
steerCmd = 0.0
throttleCmd = 0.0
brakeCmd = 0.0
is_reverse = False

try:
    while True:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == STEER:
                    steerCmd = event.value
                elif event.axis == THROTTLE:
                    throttleCmd = event.value
                elif event.axis == BRAKE:
                    brakeCmd = event.value
            elif event.type == pygame.JOYBUTTONDOWN:
                if REVERSE == event.button:
                    is_reverse = not is_reverse
        print( '转向:', "%.2f"% steerCmd, ' 油门:', "%.2f"%throttleCmd,'  刹车:',"%.2f"%brakeCmd,'  倒车:',is_reverse )
        time.sleep(0.1)

except KeyboardInterrupt:
    print("程序结束。")
finally:
    pygame.quit()