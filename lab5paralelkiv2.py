import cv2
import time
import argparse
from ultralytics import YOLO
from multiprocessing import Process, Queue
from queue import Empty

MODEL_PATH = "yolov8s-pose.pt"

class CameraCaptureRALL:
    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise IOError("Не удалось открыть камеру")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def __del__(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def process_frame(model, frame):
    result = model(frame, verbose=False)[0]
    return result.plot(boxes=False, labels=False)

def realtime_single():
    model = YOLO(MODEL_PATH)
    cam = CameraCaptureRALL()
    print("[INFO] Single-thread camera mode. Press 'q' to quit.")

    fps_list = []

    while True:
        start = time.time()
        ret, frame = cam.cap.read()
        if not ret:
            break
        processed = process_frame(model, frame)
        end = time.time()

        fps = 1 / (end - start)
        fps_list.append(fps)

        cv2.putText(processed, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-time Pose Estimation [Single]", processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    avg_fps = sum(fps_list) / len(fps_list)
    print(f"[INFO] Средний FPS (single): {avg_fps:.2f}")

def worker_process(q_in: Queue, q_out: Queue):
    model = YOLO(MODEL_PATH)
    while True:
        frame = q_in.get()
        if frame is None:
            break
        result = process_frame(model, frame)
        q_out.put(result)

def realtime_multi(num_workers=2):
    cam = CameraCaptureRALL()
    print(f"[INFO] Multi-process camera mode ({num_workers} workers). Press 'q' to quit.")

    q_in = Queue()
    q_out = Queue()
    processes = [Process(target=worker_process, args=(q_in, q_out)) for _ in range(num_workers)]
    for p in processes:
        p.start()

    fps_list = []

    try:
        while True:
            start = time.time()
            ret, frame = cam.cap.read()
            if not ret:
                break

            q_in.put(frame)

            try:
                processed = q_out.get(timeout=1)
                end = time.time()
                fps = 1 / (end - start)
                fps_list.append(fps)

                cv2.putText(processed, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Real-time Pose Estimation [Multi]", processed)
            except Empty:
                print("[WARN] Пропуск кадра")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for _ in processes:
            q_in.put(None)
        for p in processes:
            p.join()

        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        print(f"[INFO] Средний FPS (multi): {avg_fps:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Real-time YOLOv8 Pose Estimation from Camera")
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], required=True,
                        help="Режим: 'single' или 'multi'")
    parser.add_argument('--workers', type=int, default=2,
                        help="Количество процессов (только для 'multi')")
    args = parser.parse_args()

    if args.mode == 'single':
        realtime_single()
    else:
        realtime_multi(args.workers)

if __name__ == "__main__":
    main()
