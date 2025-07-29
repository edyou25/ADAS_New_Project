import pyttsx3
import time

def speak_text(text, lang='zh'):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    selected_voice_id = voices[0].id

    if lang == 'en':
        for v in voices:
            langs = []
            for l in v.languages:
                if isinstance(l, bytes):
                    langs.append(l.decode('utf-8').lower())
                else:
                    langs.append(l.lower())
            if any('en' in l for l in langs):
                selected_voice_id = v.id
                break
    elif lang == 'zh':
        for v in voices:
            langs = []
            for l in v.languages:
                if isinstance(l, bytes):
                    langs.append(l.decode('utf-8').lower())
                else:
                    langs.append(l.lower())
            if any('zh' in l for l in langs):
                selected_voice_id = v.id
                break

    engine.setProperty('voice', selected_voice_id)
    engine.say(text)
    engine.runAndWait()

def play_voice(condition):
    if condition == 'hello':
        speak_text("你好，欢迎使用本系统", lang='zh')
        speak_text("Hello, welcome to the system.", lang='en')
    elif condition == 'bye':
        speak_text("再见，感谢使用", lang='zh')
        speak_text("Goodbye, thank you for using.", lang='en')
    elif condition == 'danger_ahead':
        speak_text("前方危险", lang='zh')
        speak_text("Danger ahead!", lang='en')
    elif condition == 'pedestrian_left':
        speak_text("左侧有行人", lang='zh')
        speak_text("Pedestrian on the left.", lang='en')
    elif condition == 'motorcycle_right':
        speak_text("右侧有摩托车", lang='zh')
        speak_text("Motorcycle on the right.", lang='en')
    elif condition == 'fast_vehicle_behind':
        speak_text("后方有快速车辆", lang='zh')
        speak_text("Fast vehicle approaching from behind.", lang='en')
    else:
        speak_text("未知情况", lang='zh')
        speak_text("Unknown situation.", lang='en')

# 所有语音情况
conditions = [
    'hello',
    'bye',
    'danger_ahead',
    'pedestrian_left',
    'motorcycle_right',
    'fast_vehicle_behind',
    'unknown'
]

for cond in conditions:
    print(f"\n正在播放：{cond}")
    play_voice(cond)
    time.sleep(1)  # 可选：每条语音之间停顿1秒