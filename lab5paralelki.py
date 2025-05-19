import cv2
import argparse
import time
from ultralytics import YOLO
from multiprocessing import Process, Queue
from queue import Empty

MODEL_PATH = "yolov8s-pose.pt"
DEFAULT_VIDEO_PATH = "input.mp4"
DEFAULT_OUTPUT_PATH = "output.mp4"

class VideoCaptureRALL:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        if self.cap:
            self.cap.release()

def process_frame(model, frame):
    result = model(frame, verbose=False)[0]
    return result.plot()

def run_single_thread():
    model = YOLO(MODEL_PATH)
    cap = VideoCaptureRALL(DEFAULT_VIDEO_PATH)
    out = cv2.VideoWriter(DEFAULT_OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), cap.fps, (cap.width, cap.height))

    start = time.time()
    while True:
        ret, frame = cap.cap.read()
        if not ret:
            break
        processed = process_frame(model, frame)
        out.write(processed)
    end = time.time()

    out.release()
    return end - start

def worker(q_in: Queue, q_out: Queue):
    model = YOLO(MODEL_PATH)
    while True:
        task = q_in.get()
        if task is None:
            break
        idx, frame = task
        processed = process_frame(model, frame)
        q_out.put((idx, processed))

def run_multi_process(num_workers):
    cap = VideoCaptureRALL(DEFAULT_VIDEO_PATH)
    q_in = Queue()
    q_out = Queue()
    out = cv2.VideoWriter(DEFAULT_OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), cap.fps, (cap.width, cap.height))

    workers = [Process(target=worker, args=(q_in, q_out)) for _ in range(num_workers)]
    for w in workers:
        w.start()

    index = 0
    while True:
        ret, frame = cap.cap.read()
        if not ret:
            break
        q_in.put((index, frame))
        index += 1

    frame_count = index

    for _ in workers:
        q_in.put(None)

    results = {}
    for _ in range(frame_count):
        try:
            idx, processed = q_out.get(timeout=30)
            results[idx] = processed
        except Empty:
            print("Время ожидания кадра истекло.")
            break

    for i in sorted(results.keys()):
        out.write(results[i])

    for w in workers:
        w.join()
    out.release()

    return time.time() - start_time

def main():
    parser = argparse.ArgumentParser(description="Inference YOLOv8s-Pose")
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], required=True, help='Режим обработки: single или multi')
    parser.add_argument('--workers', type=int, default=2, help='Количество процессов (для multi)')
    args = parser.parse_args()

    print(f"[INFO] Начинаем обработку в режиме: {args.mode}")

    if args.mode == 'single':
        duration = run_single_thread()
    else:
        global start_time
        start_time = time.time()
        duration = run_multi_process(args.workers)

    print(f"[INFO] Обработка завершена. Время выполнения: {duration:.2f} секунд.")
    print(f"[INFO] Выходной файл сохранён как: {DEFAULT_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
