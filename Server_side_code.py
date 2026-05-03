#Server Side(Laptop)
import cv2
import socket
import struct
import threading
import numpy as np
import queue
import time
from ultralytics import YOLO

MODEL_PATH = r"D:\lane_detection\best_traffic_v2.pt"#replace
CONF = 0.6
IOU = 0.5
SERVER_IP = "0.0.0.0"
SERVER_PORT = 9999
TCP_RCVBUF = 131_072

print("[SERVER] Loading YOLO model…")
try:
    model = YOLO(MODEL_PATH)
    print("[SERVER] Model loaded")
except Exception as e:
    print(f"[SERVER] ERROR loading model: {e}"); exit(1)

def recvall(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def yolo_worker(yolo_queue, client_socket, stop_event):
    frames_done = 0
    last_command = "GO"
    infer_total = 0.0
    diag_last   = time.time()
    diag_count  = 0

    while not stop_event.is_set():
        try:
            frame_count, frame = yolo_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        t0      = time.perf_counter()
        command = "GO"

        try:
            results = model(frame, conf=CONF, iou=IOU, verbose=False)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_name = model.names[int(box.cls[0])].lower()
                    conf_val = float(box.conf[0])

                    
                    if "stop" in cls_name:
                        command = "STOP"
                        print(f"[YOLO] #{frame_count}: STOP SIGN ({conf_val:.2f})")
                        break
                    elif "red" in cls_name:
                        command = "STOP"
                        print(f"[YOLO] #{frame_count}: RED LIGHT ({conf_val:.2f})")
                        break
                    
                    elif "green" in cls_name and command == "GO":
                        command = "GREEN"
                        print(f"[YOLO] #{frame_count}: GREEN ({conf_val:.2f})")

        except Exception as e:
            print(f"[YOLO] Inference error frame {frame_count}: {e}")
            command = last_command

        last_command = command
        infer_ms     = (time.perf_counter() - t0) * 1000
        infer_total += infer_ms
        diag_count  += 1

        
        try:
            resp = (struct.pack('I', frame_count) +
                    command.encode().ljust(30, b'\x00'))
            client_socket.sendall(resp)
            frames_done += 1
            print(f"[SEND] #{frame_count}: {command} ({infer_ms:.1f} ms)")
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[YOLO] Send failed — client disconnected")
            stop_event.set()
            break

        now = time.time()
        if now - diag_last >= 10.0 and diag_count:
            avg_ms = infer_total / diag_count
            fps    = diag_count  / (now - diag_last)
            print(f"[DIAG] {fps:.1f} fps processed | avg inference {avg_ms:.1f} ms "
                  f"| total done {frames_done}")
            infer_total = 0.0; diag_count = 0; diag_last = now

    print(f"[YOLO] Worker stopped ({frames_done} frames processed)")


def handle_client(client_socket, client_address):
    print(f"[CLIENT] Connected from {client_address}")
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    yolo_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    worker = threading.Thread(target=yolo_worker,
                              args=(yolo_queue, client_socket, stop_event),
                              daemon=True)
    worker.start()

    rcvd   = 0
    drops  = 0
    t_last = time.time()

    try:
        while not stop_event.is_set():

            header = recvall(client_socket, 8)
            if header is None:
                print("[CLIENT] Pi disconnected")
                break

            frame_count, jpeg_size = struct.unpack('II', header)

            jpeg_data = recvall(client_socket, jpeg_size)
            if jpeg_data is None:
                print(f"[CLIENT] Incomplete frame #{frame_count}")
                break

            frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[CLIENT] Decode error frame #{frame_count}")
                continue

            rcvd += 1

            if yolo_queue.full():
                try: yolo_queue.get_nowait(); drops += 1
                except queue.Empty: pass
            try:
                yolo_queue.put_nowait((frame_count, frame))
            except queue.Full:
                drops += 1

            now = time.time()
            if now - t_last >= 10.0:
                fps = rcvd / (now - t_last)
                print(f"[RECV] {fps:.1f} fps received | {drops} dropped | "
                      f"last frame #{frame_count}")
                rcvd = drops = 0; t_last = now

    except (ConnectionResetError, OSError) as e:
        print(f"[CLIENT] Recv error: {e}")
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        client_socket.close()
        print(f"[CLIENT] Disconnected from {client_address}")

def start_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TCP_RCVBUF)
    srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        srv.bind((SERVER_IP, SERVER_PORT))
        srv.listen(1)
        srv.settimeout(1.0)
        print(f"[SERVER] Listening {SERVER_IP}:{SERVER_PORT}")
        print(f"[SERVER] Model: {MODEL_PATH}")
        print(f"[SERVER] conf={CONF}  iou={IOU}  rcvbuf={TCP_RCVBUF//1024}KB")
        print("[SERVER] Waiting for Pi…")

        while True:
            try:
                csock, caddr = srv.accept()
                threading.Thread(target=handle_client,
                                 args=(csock, caddr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[SERVER] Accept error: {e}")

    except Exception as e:
        print(f"[SERVER] Bind error: {e}")
    finally:
        srv.close()
        print("[SERVER] Stopped")

if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down…")


