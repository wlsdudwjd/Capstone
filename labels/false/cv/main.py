import RPi.GPIO as GPIO
import time
import requests
import board
import busio
import adafruit_ssd1306
import subprocess
import threading
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

PIR = 4
LED = 17
BUZZER = 27
TRIG = 23
ECHO = 24
BUTTON = 22

BOT_TOKEN="8904189049:AAHdOjX6Kfv2iAgCvAl4rvMeqWDHI_VYn2o"
CHAT_ID="8719981857"

GPIO.setup(PIR, GPIO.IN)
GPIO.setup(LED, GPIO.OUT)
GPIO.setup(BUZZER, GPIO.OUT)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwm = GPIO.PWM(BUZZER, 262)

i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)

image = Image.new("1", (128, 64))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

def show_oled(line1="", line2="", line3="", line4=""):
        draw.rectangle((0, 0, 128, 64), outline=0, fill=0)

        draw.text((0, 0), line1, font=font, fill=225)
        draw.text((0, 16), line2, font=font, fill=225)
        draw.text((0, 32), line3, font=font, fill=225)
        draw.text((0, 48), line4, font=font, fill=225)

        oled.image(image)
        oled.show()

def send_telegram(message):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
                "chat_id": CHAT_ID,
                "text": message
                }
        try:

                response = requests.post(url, data=data, timeout=5)

                if response.status_code == 200:
                        print("telegram_success")
                else:
                        print("telegram_fail")
                        print(response.text)
        except Exception as e:
                print("telegram_error", e)

def take_photo():
        filename = datetime.now().strftime("warning_%Y%m%d_%H%M%S.jpg")

        try:
                subprocess.run(
                        ["rpicam-still", "-o", filename, "--timeout", "1000"],
                        check=True
                )
                print(f"save photo: {filename}")
                return filename

        except Exception as e:
                print("rpicam-still fail:", e)

                try:
                        subprocess.run(
                                ["libcamera-still", "-o", filename, "--timeout", "1000"],
                                check=True
                        )
                        print(f"save photo success: {filename}")
                        return filename

                except Exception as e:
                        print("take photo fail", e)
                        return None

def send_telegram_photo(photo_path, caption=""):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

        try:
                with open(photo_path, "rb") as photo:
                        data = {
                                "chat_id": CHAT_ID,
                                "caption": caption
                        }

                        files = {
                                "photo": photo
                        }

                        response = requests.post(url, data=data, files=files, timeout=10)

                if response.status_code == 200:
                        print("telegram_photo_send_success")
                else:
                        print("telegram_photo_send_fail")
                        print(response.text)
        except Exception as e:
                print("telegram photo error:", e)

def alert_task(distance):
        message = f"""
                        [SMART TURRET WARNING]

                        WARNING!

                        DISTANCE: {distance:.1f}cm
                        STATUS: DANGER
                        MODE: ON
                        """
        send_telegram(message)
        photo_path = take_photo()

        if photo_path is not None:
                send_telegram_photo(
                        photo_path,
                        caption=f"motion detect / distance: {distance:.1f}cm"
                )


def beep(freq=523, duration=0.2, duty=20):
        pwm.start(duty)
        pwm.ChangeFrequency(freq)
        time.sleep(duration)
        pwm.stop()

def warning_beep(level):
        if level == 1:
                beep(523, 0.15)
        elif level == 2:
                for _ in range(2):
                        beep(700, 0.12)
                        time.sleep(0.1)
        elif level == 3:
                for _ in range(3):
                        beep(900, 0.1)
                        time.sleep(0.05)

def get_distance():
        GPIO.output(TRIG, False)
        time.sleep(0.05)

        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        start_time = time.time()
        timeout = start_time + 0.03

        while GPIO.input(ECHO) == 0:
                start_time = time.time()
                if start_time > timeout:
                        return None

        stop_time = time.time()
        timeout = stop_time + 0.03

        while GPIO.input(ECHO) == 1:
                stop_time = time.time()
                if stop_time > timeout:
                        return None

        elapsed = stop_time - start_time
        distance = elapsed * 34300 / 2

        return distance

print("hello")
show_oled("smart turret", "system start", "BUTTON MODE", "")
time.sleep(2)

armed = True
last_alert_time = 0
ALERT_COOLDOWN = 10
last_button_state = GPIO.HIGH
last_display_mode =""
print("start")
show_oled("smart turret", "mode: WATCHING", "motion: no", "dist: --cm")


try:
        while True:
                button_state = GPIO.input(BUTTON)

                if last_button_state == GPIO.HIGH and button_state == GPIO.LOW:
                        armed = not armed

                        if armed:
                                print("detected mode on")
                                show_oled("SMART TURRET", "MODE: WATCHING", "SYSTEM: ON", "")
                                beep(700, 0.1)
                        else:
                                print("detected mode off")
                                GPIO.output(LED, GPIO.LOW)
                                show_oled("SMART TURRET", "MODE: OFF", "SYSTEM: STOP", "")
                                beep(300, 0.1)

                        time.sleep(0.3)

                last_button_state = button_state

                if not armed:
                        GPIO.output(LED, GPIO.LOW)
                        time.sleep(0.1)
                        continue

                motion = GPIO.input(PIR)

                if motion == 1:
                        distance = get_distance()
                        GPIO.output(LED, GPIO.HIGH)

                        if distance is None:
                                print("motion detected / distance fail")
                                show_oled("motion detected", "dist: error", "status: check", "MODE: ON")
                                warning_beep(1)

                        elif distance > 100:
                                print(f"motion detect / distance: {distance:.1f}cm / status : detect")
                                show_oled("motion detected",
                                                        f"dist: {distance:.1f} cm",
                                                        "status: motion",
                                                        "level: 1"
                                                        )
                                warning_beep(1)

                        elif distance > 50:
                                print(f"motion detect / distance: {distance:.1f}cm / status : near")
                                show_oled("object near",
                                                        f"dist: {distance:.1f} cm",
                                                        "status: near",
                                                        "level: 2"
                                                        )
                                warning_beep(2)

                        else:
                                print(f"motion detect / distance: {distance:.1f}cm / status : danger")
                                show_oled("warning!",
                                                        f"dist: {distance:.1f} cm",
                                                        "status: danger",
                                                        "level: 3"
                                                        )


                                warning_beep(3)

                                now = time.time()

                                if now - last_alert_time > ALERT_COOLDOWN:
                                        last_alert_time = now

                                        threading.Thread(
                                                target=alert_task,
                                                args=(distance,),
                                                daemon=True
                                        ).start()
                        time.sleep(0.5)

                else:
                        GPIO.output(LED, GPIO.LOW)

                        show_oled("smart turret", "mode: watching", "motion: no", "dist: --cm")
                        last_display_mode = "WATCHING"

                        print("no motion")
                        time.sleep(0.5)

except KeyboardInterrupt:
        print("end")
        show_oled("smart turret", "system off", "", "")
        time.sleep(1)

finally:
        pwm.stop()
        oled.fill(0)
        oled.show()
        GPIO.cleanup()