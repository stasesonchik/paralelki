from abc import ABC, abstractmethod
import threading
import queue
import time
import cv2
import logging


class Sensor(ABC):
    @abstractmethod
    def get(self):
        pass


class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self, camera_name, resolution=(1280, 720)):
        self.camera_name = camera_name
        self.resolution = resolution
        self.cap = cv2.VideoCapture(camera_name)

        if not self.cap.isOpened():
            logging.error(f"Ошибка открытия камеры {camera_name}")
            raise Exception("Ошибка открытия камеры")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    def get(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Ошибка получения кадра с камеры")
            return None
        return frame

    def close(self):
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.close()
        logging.info(f"Камера {self.camera_name} освобождена")


class WindowImage:
    def __init__(self, display_frequency):
        self.display_frequency = display_frequency
        self.window_name = "Sensor Data"

    def show(self, img, sensor0_data, sensor1_data, sensor2_data):
        if img is not None:
            cv2.putText(img, f"Sensor0: {sensor0_data}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Sensor1: {sensor1_data}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img, f"Sensor2: {sensor2_data}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(self.window_name, img)

    def close(self):
        cv2.destroyWindow(self.window_name)

    def __del__(self):
        self.close()
        logging.info("Окно отображения закрыто")


def sensor_worker(sensor, queue_sensor, stop_event, lock, sync_interval=0.1):
    last_time = time.time()
    while not stop_event.is_set():
        data = sensor.get()
        timestamp = time.time()
        with lock:
            if queue_sensor.qsize() >= 1:
                queue_sensor.get_nowait()
            queue_sensor.put((data, timestamp))

        time_to_wait = sync_interval - (time.time() - last_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        last_time = time.time()


def try_get_new_data(q, last_value):
    try:
        data, _ = q.get_nowait()
        return data
    except queue.Empty:
        return last_value


def main(camera_name, resolution, display_frequency):
    cam_sensor = SensorCam(camera_name, resolution)

    sensor0 = SensorX(0.01)
    sensor1 = SensorX(0.1)
    sensor2 = SensorX(1)

    window = WindowImage(display_frequency)

    cam_queue = queue.Queue(maxsize=1)
    sensor0_queue = queue.Queue(maxsize=1)
    sensor1_queue = queue.Queue(maxsize=1)
    sensor2_queue = queue.Queue(maxsize=1)

    stop_event = threading.Event()
    lock = threading.Lock()

    sensor_threads = [
        threading.Thread(target=sensor_worker, args=(cam_sensor, cam_queue, stop_event, lock), daemon=True),
        threading.Thread(target=sensor_worker, args=(sensor0, sensor0_queue, stop_event, lock), daemon=True),
        threading.Thread(target=sensor_worker, args=(sensor1, sensor1_queue, stop_event, lock), daemon=True),
        threading.Thread(target=sensor_worker, args=(sensor2, sensor2_queue, stop_event, lock), daemon=True),
    ]

    for thread in sensor_threads:
        thread.start()

    sensor0_data = 0
    sensor1_data = 0
    sensor2_data = 0

    try:
        while True:
            cam_data = try_get_new_data(cam_queue, None)

            if cam_data is not None:
                img = cam_data.copy()
                sensor0_data = try_get_new_data(sensor0_queue, sensor0_data)
                sensor1_data = try_get_new_data(sensor1_queue, sensor1_data)
                sensor2_data = try_get_new_data(sensor2_queue, sensor2_data)

                window.show(img, sensor0_data, sensor1_data, sensor2_data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Нажата клавиша 'q', завершение программы")
                break

            time.sleep(1 / display_frequency)

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
    finally:
        stop_event.set()
        time.sleep(0.1)
        cam_sensor.close()
        window.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import os

    os.makedirs("log", exist_ok=True)
    logging.basicConfig(filename='log/app.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Управление сенсорами")
    parser.add_argument('camera_name', help="Имя камеры в системе (например, 0)")
    parser.add_argument('--resolution', type=str, default='1280x720', help="Разрешение камеры")
    parser.add_argument('--display_frequency', type=int, default=30, help="Частота обновления окна")

    args = parser.parse_args()
    resolution = tuple(map(int, args.resolution.split('x')))
    main(int(args.camera_name), resolution, args.display_frequency)
