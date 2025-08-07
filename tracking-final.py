from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time
import datetime

JARAK_REF = 160.0
LEBAR_PIKSEL_REF = 140.0
TARGET_AREA = 65000
PID_P_YAW, PID_D_YAW = 0.22, 0.18
PID_P_UD = 0.3
PID_P_FB_FORWARD, PID_P_FB_BACKWARD = 0.0008, 0.00015
TOLERANCE_X, TOLERANCE_Y = 35, 35
TOLERANCE_AREA = 8000
MAX_SPEED, MIN_SPEED = 30, -30
MAX_ALTITUDE_CM = 160
SEARCH_TIMEOUT = 2.0
MIN_BATTERY_TAKEOFF = 20


tello = Tello()
area_history = []
MAX_AREA_HISTORY = 5
previous_error_x = 0
last_target_seen_time = time.time()

def draw_stats(frame, tello, rc_commands):
    """Menampilkan status drone DAN perintah RC yang dihitung."""
    lr, fb, ud, yaw = rc_commands
    try:
        altitude_cm = tello.get_height()
        battery = tello.get_battery()
        stats_text = f"Alt: {altitude_cm} cm | Bat: {battery}%"
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        rc_text = f"RC Out -> FB: {fb}, Yaw: {yaw}, UD: {ud}"
        cv2.putText(frame, rc_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2)

    except Exception as e:
        cv2.putText(frame, f"Sensor Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def track_person(frame, results, frame_center_x, frame_center_y):
    global previous_error_x, last_target_seen_time
    target_box = None
    max_area = 0;
    max_conf = 0.0;
    distance = 0.0;
    current_width = 0
    if results and results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            if cls_id == TARGET_PERSON_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area;
                    target_box = (x1, y1, x2, y2)
                    max_conf = float(box.conf[0].cpu().numpy());
                    current_width = x2 - x1
    lr, fb, ud, yaw = 0, 0, 0, 0
    if target_box:
        last_target_seen_time = time.time()
        x1, y1, x2, y2 = target_box;
        box_center_x = (x1 + x2) // 2;
        box_center_y = (y1 + y2) // 2
        area_history.append(max_area)
        if len(area_history) > MAX_AREA_HISTORY: area_history.pop(0)
        smoothed_area = sum(area_history) / len(area_history)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        info_text = f"Area: {int(smoothed_area)}";
        cv2.putText(frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if current_width > 0:
            distance = (LEBAR_PIKSEL_REF * JARAK_REF) / current_width
            cv2.putText(frame, f"Jarak: {distance:.2f} cm", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                        2)
        error_x = box_center_x - frame_center_x
        if abs(error_x) > TOLERANCE_X: yaw = (PID_P_YAW * error_x) + (PID_D_YAW * (error_x - previous_error_x))
        previous_error_x = error_x
        vertical_target_y = frame_center_y + 40;
        error_y = vertical_target_y - box_center_y
        if abs(error_y) > TOLERANCE_Y: ud = error_y * PID_P_UD
        if smoothed_area < (TARGET_AREA - TOLERANCE_AREA):
            fb = (TARGET_AREA - smoothed_area) * PID_P_FB_FORWARD
        elif smoothed_area > (TARGET_AREA + TOLERANCE_AREA):
            fb = (TARGET_AREA - smoothed_area) * PID_P_FB_BACKWARD
    else:
        if time.time() - last_target_seen_time > SEARCH_TIMEOUT:
            yaw = 25
        else:
            lr, fb, ud, yaw = 0, 0, 0, 0
        previous_error_x = 0
    return lr, fb, ud, yaw, max_conf, distance


def send_rc_command(lr, fb, ud, yaw):
    lr = int(max(min(lr, MAX_SPEED), MIN_SPEED));
    fb = int(max(min(fb, MAX_SPEED), MIN_SPEED))
    ud = int(max(min(ud, MAX_SPEED), MIN_SPEED));
    yaw = int(max(min(yaw, MAX_SPEED), MIN_SPEED))
    tello.send_rc_control(lr, fb, ud, yaw)
    return lr, fb, ud, yaw

try:
    print("Menghubungkan ke Tello...")
    tello.connect()
    battery = tello.get_battery()
    print(f"Terhubung! Baterai: {battery}%")
    if battery < MIN_BATTERY_TAKEOFF: raise Exception("Baterai di bawah ambang batas aman.")
    tello.streamon()
    model = YOLO("yolov8n.pt")
    TARGET_PERSON_CLASS_ID = 0
    print("\n--- PERINGATAN: DRONE AKAN TAKEOFF OTOMATIS ---")
    time.sleep(2)
    tello.takeoff()
    time.sleep(1)
    tello.send_rc_control(0, 0, 25, 0)
    time.sleep(2)
    print("Stabilisasi selesai.")
    frame_read = tello.get_frame_read()
    print("--- MEMULAI PELACAKAN OTOMATIS ---")

    while True:
        frame = frame_read.frame
        if frame is None: time.sleep(0.1); continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = frame.copy()
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        results = model(annotated_frame, verbose=False)
        lr, fb, ud, yaw, confidence, est_distance = track_person(annotated_frame, results, frame_center_x,
                                                                 frame_center_y)

        if tello.get_height() > MAX_ALTITUDE_CM: ud = -20

        final_lr, final_fb, final_ud, final_yaw = send_rc_command(lr, fb, ud, yaw)

        draw_stats(annotated_frame, tello, (final_lr, final_fb, final_ud, final_yaw))

        cv2.imshow("Tello Stable Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram dihentikan oleh pengguna (Ctrl+C).")
except Exception as e:
    print(f"Terjadi error: {e}")
finally:
    print("--- PROSES PENDARATAN OTOMATIS ---")
    if 'tello' in locals() and tello.is_flying:
        try:
            tello.land()
        except Exception as e_land:
            print(f"Gagal mendarat normal: {e_land}."); tello.send_rc_control(0, 0, 0, 0)
    if 'tello' in locals(): tello.streamoff()
    cv2.destroyAllWindows()
    print("Program selesai.")