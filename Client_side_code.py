#Client side (Raspberry Pi code)

import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
import socket
import struct
import threading
import queue

LAPTOP_IP = "" #Your laptop IP
LAPTOP_PORT = 9999
ENA, IN1, IN2 = 11, 13, 15
ENB, IN3, IN4 = 16, 37, 38
CAMERA_ID = 0
SPEED = 40
TURN_SPEED = 35
S_MAX = 50
V_MIN = 160
MIN_WHITE_PIXELS = 150
NEAR_TOP, NEAR_BOT = 0.55, 1.00
FAR_TOP, FAR_BOT = 0.25, 0.55
NEAR_WEIGHT = 0.70
FAR_WEIGHT = 0.30
DEAD_ZONE = 0.12
LANE_LOST_PCT = 10
ARRIVAL_CONFIRM_FRAMES = 20
MIN_FRAMES_BEFORE_ARRIVAL = 100

OBSTACLE_NAMES = ["STOP"] 
FRAME_SEND_INTERVAL = 2
JPEG_QUALITY = 50
CLEAR_CONFIRM_FRAMES = 8
RESUME_DELAY = 1.2

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
for pin in [IN1, IN2, IN3, IN4]:
    GPIO.setup(pin, GPIO.OUT)
for pin in [ENA, ENB]:
    GPIO.setup(pin, GPIO.OUT)
pwm_a = GPIO.PWM(ENA, 50); pwm_a.start(0)
pwm_b = GPIO.PWM(ENB, 50); pwm_b.start(0)

def motors_forward():
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def motors_left():
    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(TURN_SPEED)
    pwm_b.ChangeDutyCycle(TURN_SPEED)

def motors_right():
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(TURN_SPEED)
    pwm_b.ChangeDutyCycle(TURN_SPEED)

def motors_stop():
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

def camera_thread(camera_queue, stop_event):
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("[CAM] ERROR: Camera not found!")
        stop_event.set()
        return

    print("[CAM] Camera OK")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if camera_queue.full():
            try:
                camera_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            camera_queue.put_nowait(frame)
        except queue.Full:
            pass

    cap.release()
    print("[CAM] Camera thread stopped")

def get_white_mask(frame):
    small = cv2.resize(frame, (160, 120))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([0, 0, V_MIN]),
                       np.array([180, S_MAX, 255]))
    return small, mask

def get_roi_offset(mask, top_frac, bot_frac, min_pixels):
    h, w = mask.shape
    roi = mask[int(h * top_frac):int(h * bot_frac), :]
    pixels = cv2.countNonZero(roi)
    if pixels < min_pixels:
        return 0.0, pixels
    M = cv2.moments(roi)
    if M["m00"] == 0:
        return 0.0, pixels
    return (M["m10"] / M["m00"] - w / 2.0) / (w / 2.0), pixels

def get_lane_info(frame):
    small, mask = get_white_mask(frame)
    h, w = mask.shape
    white_pct = cv2.countNonZero(mask) / float(h * w) * 100.0
    near_off, near_px = get_roi_offset(
        mask, NEAR_TOP, NEAR_BOT, MIN_WHITE_PIXELS)
    far_off,  far_px  = get_roi_offset(
        mask, FAR_TOP,  FAR_BOT,  MIN_WHITE_PIXELS // 2)

    if near_px >= MIN_WHITE_PIXELS and far_px >= MIN_WHITE_PIXELS // 2:
        blend = NEAR_WEIGHT * near_off + FAR_WEIGHT * far_off
    elif near_px >= MIN_WHITE_PIXELS:
        blend = near_off
    elif far_px >= MIN_WHITE_PIXELS // 2:
        blend = far_off
    else:
        blend = 0.0

    return white_pct, near_off, far_off, blend, small, mask, near_px, far_px

def show_frame(small, mask, action, white_pct, state, blend, yolo_cmd):
    mc = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    h, w = mc.shape[:2]
    for frac, color in [
        (NEAR_TOP, (0, 255, 0)),
        (FAR_TOP,  (255, 165, 0)),
        (FAR_BOT,  (255, 165, 0))
    ]:
        cv2.line(mc, (0, int(h * frac)), (w, int(h * frac)), color, 1)
    cv2.line(mc, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)

    col = (255, 100, 0)  if action == "LEFT"    else \
          (0, 100, 255)  if action == "RIGHT"   else \
          (0, 220, 0)    if action == "FORWARD" else \
          (0, 0, 255)    if "STOP" in action    else \
          (180, 180, 180)

    out = np.hstack((small, mc))

    texts = [
        (action,                        20, col),
        ("White:%d%%" % int(white_pct), 38, (255, 255, 255)),
        ("State:" + state,              56, (200, 200, 0)),
        ("Blend:%.2f" % blend,          74, (180, 255, 180)),
        ("YOLO:" + yolo_cmd,            92, (255, 200, 0)),
    ]
    for txt, y, c in texts:
        cv2.putText(out, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    cv2.imshow("LaneCar+YOLO", out)

def network_thread(frame_queue, command_queue, stop_event):
    frames_sent = 0
    frames_dropped = 0
    print(f"[NET] Will connect to {LAPTOP_IP}:{LAPTOP_PORT}")

    while not stop_event.is_set():

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET,  socket.SO_KEEPALIVE, 1)

        try:
            sock.settimeout(5.0)
            sock.connect((LAPTOP_IP, LAPTOP_PORT))
            sock.settimeout(None)
            print(f"[NET] Connected to {LAPTOP_IP}:{LAPTOP_PORT}")
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"[NET] Connect failed: {e} — retrying in 2s")
            sock.close()
            if stop_event.is_set():
                break
            time.sleep(2.0)
            continue

        recv_buf = b''

        try:
            while not stop_event.is_set():

                try:
                    fc, frame = frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                ret, jpeg = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if not ret:
                    print("[NET] JPEG encode failed, skipping frame")
                    continue

                data   = jpeg.tobytes()
                header = struct.pack('II', fc, len(data))
                packet = header + data

                sock.settimeout(10.0)
                try:
                    sock.sendall(packet)
                    frames_sent += 1
                    print(f"[NET] Sent frame #{fc} ({len(data)//1024}KB)")
                except (socket.timeout, BrokenPipeError,
                        ConnectionResetError, OSError) as e:
                    print(f"[NET] Send FAILED frame #{fc}: {e}")
                    frames_dropped += 1
                    break

                sock.settimeout(1.0)
                try:
                    data_in = sock.recv(4096)
                    if data_in:
                        recv_buf += data_in

                        while len(recv_buf) >= 34:
                            pkt      = recv_buf[:34]
                            recv_buf = recv_buf[34:]

                            pkt_fc  = struct.unpack('I', pkt[:4])[0]
                            cmd_raw = pkt[4:34]
                            cmd     = cmd_raw.decode(
                                'utf-8', errors='ignore'
                            ).rstrip('\x00').strip()

                            if cmd:
                                if command_queue.full():
                                    try:
                                        command_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                try:
                                    command_queue.put_nowait((pkt_fc, cmd))
                                    print(f"[NET] Received cmd: {cmd} "
                                          f"(frame #{pkt_fc})")
                                except queue.Full:
                                    pass
                    else:
                        print("[NET] Server closed connection")
                        break

                except socket.timeout:
                    pass
                except (ConnectionResetError, OSError) as e:
                    print(f"[NET] Recv error: {e}")
                    break

        except Exception as e:
            print(f"[NET] Unexpected error: {type(e).__name__}: {e}")

        finally:
            try:
                sock.close()
            except:
                pass
            print(f"[NET] Disconnected. Sent={frames_sent} "
                  f"Dropped={frames_dropped}")

        if stop_event.is_set():
            break

        print("[NET] Reconnecting in 2s...")
        time.sleep(2.0)

    print("[NET] Network thread stopped")

def main():
    print(f"[MAIN] Speed={SPEED} Turn={TURN_SPEED}")
    print("[MAIN] Starting in 3s...")
    time.sleep(3)
    camera_queue = queue.Queue(maxsize=1)
    frame_queue  = queue.Queue(maxsize=2)
    cmd_queue    = queue.Queue(maxsize=10)
    stop_event   = threading.Event()

    cam_thread = threading.Thread(
        target=camera_thread,
        args=(camera_queue, stop_event),
        daemon=True
    )
    net_thread = threading.Thread(
        target=network_thread,
        args=(frame_queue, cmd_queue, stop_event),
        daemon=True
    )

    cam_thread.start()
    time.sleep(1.0)
    net_thread.start()

    print("[MAIN] GO!")

    FOLLOW   = "FOLLOW"
    STOP_OBS = "STOP_OBSTACLE"
    ARRIVED  = "ARRIVED"

    state               = FOLLOW
    action              = "starting"
    frame_count         = 0
    yolo_cmd            = "GO"
    arrival_count       = 0
    clear_frame_count   = 0
    obstacle_cleared_at = 0
    detected_object     = ""

    white_pct = 0.0
    blend     = 0.0
    small     = None
    mask      = None
    near_px   = 0
    far_px    = 0

    try:
        while True:

            try:
                frame = camera_queue.get(timeout=0.2)
            except queue.Empty:
                print("[MAIN] Waiting for camera...")
                continue

            frame_count += 1

            (white_pct, near_off, far_off, blend,
             small, mask, near_px, far_px) = get_lane_info(frame)

            if frame_count % FRAME_SEND_INTERVAL == 0:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    frame_queue.put_nowait((frame_count, frame.copy()))
                except queue.Full:
                    pass

            latest_cmd = None
            while True:
                try:
                    latest_cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break

            if latest_cmd is not None:
                yolo_cmd = latest_cmd[1]
                print(f"[FSM] YOLO cmd: {yolo_cmd}")

            if (white_pct < LANE_LOST_PCT and
                    frame_count >= MIN_FRAMES_BEFORE_ARRIVAL):
                arrival_count += 1
            else:
                arrival_count = 0

            if yolo_cmd in OBSTACLE_NAMES:
                if state != STOP_OBS:
                    print(f"[FSM] FOLLOW → STOP_OBSTACLE ({yolo_cmd})")
                state               = STOP_OBS
                detected_object     = yolo_cmd
                arrival_count       = 0
                clear_frame_count   = 0
                obstacle_cleared_at = 0

            elif state == STOP_OBS:
                
                if yolo_cmd in ("GREEN", "GO"):
                    clear_frame_count += 1

                    if (obstacle_cleared_at == 0 and
                            clear_frame_count >= CLEAR_CONFIRM_FRAMES):
                        obstacle_cleared_at = time.time()
                        print(f"[FSM] Obstacle cleared! "
                              f"Waiting {RESUME_DELAY}s before resume")
                    elif (obstacle_cleared_at > 0 and
                          time.time() - obstacle_cleared_at >= RESUME_DELAY):
                        print("[FSM] STOP_OBSTACLE → FOLLOW")
                        state               = FOLLOW
                        detected_object     = ""
                        clear_frame_count   = 0
                        obstacle_cleared_at = 0
                else:
                    clear_frame_count   = 0
                    obstacle_cleared_at = 0

            elif (state == FOLLOW and
                  arrival_count >= ARRIVAL_CONFIRM_FRAMES):
                state = ARRIVED

            if state == FOLLOW:
                if white_pct < LANE_LOST_PCT:
                    motors_stop()
                    action = "LANE LOST"
                elif (near_px < MIN_WHITE_PIXELS and
                      far_px < MIN_WHITE_PIXELS // 2):
                    motors_stop()
                    action = "NO LANE"
                elif blend < -DEAD_ZONE:
                    motors_left()
                    action = "LEFT (%.2f)" % blend
                elif blend > DEAD_ZONE:
                    motors_right()
                    action = "RIGHT (%.2f)" % blend
                else:
                    motors_forward()
                    action = "FORWARD"

            elif state == STOP_OBS:
                motors_stop()
                cleared = (time.time() - obstacle_cleared_at
                           if obstacle_cleared_at else 0)
                action = (f"STOPPED:{detected_object} "
                          f"clr={clear_frame_count} t={cleared:.1f}s")

            elif state == ARRIVED:
                motors_stop()
                print("[MAIN] *** DESTINATION REACHED ***")
                time.sleep(0.5)
                break

            if small is not None and mask is not None:
                show_frame(small, mask, action,
                           white_pct, state, blend, yolo_cmd)

            if frame_count % 30 == 0:
                print(f"[MAIN] F={frame_count} st={state} "
                      f"wh={white_pct:.0f}% blend={blend:.2f} "
                      f"yolo={yolo_cmd} act={action}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[MAIN] Stopped by user.")
    except Exception as e:
        print(f"[MAIN] Error: {e}")
    finally:
        print("[MAIN] Shutting down...")
        stop_event.set()

        cam_thread.join(timeout=2.0)
        net_thread.join(timeout=2.0)

        motors_stop()
        pwm_a.stop()
        pwm_b.stop()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("[MAIN] Cleanup done.")

if __name__ == "__main__":
    main()
