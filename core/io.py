import time, cv2
from dataclasses import dataclass
from typing import Tuple

@dataclass
class VideoInfo:
    fps: float
    size_wh: Tuple[int,int]

class VideoReader:
    def __init__(self, path:str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    def read(self):
        return self.cap.read()
    def release(self):
        self.cap.release()
    def info(self) -> VideoInfo:
        return VideoInfo(self.fps, (self.W, self.H))

class FpsPacer:
    def __init__(self, fps: float):
        self.fps = fps
        self.dt = 1.0 / max(1e-6, fps)
        self.t0 = time.perf_counter()
    def sleep_for_frame(self, frame_idx:int):
        target = self.t0 + frame_idx * self.dt
        now = time.perf_counter()
        st = target - now
        if st > 0:
            time.sleep(st)
