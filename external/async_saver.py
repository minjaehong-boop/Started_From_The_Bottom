import cv2
import time
from multiprocessing import Process, Queue, Event
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SaverConfig:
    path: str
    fps: float
    size_wh: Tuple[int, int]  # (W, H)
    fourcc: str = "mp4v"
    max_queue: int = 2048
    drop_old_when_full: bool = False

def _saver_proc(frame_q: Queue, stop_evt: Event, cfg: SaverConfig):
    fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)
    writer = cv2.VideoWriter(cfg.path, fourcc, cfg.fps, cfg.size_wh)
    if not writer.isOpened():
        print(f"[saver] cannot open writer: {cfg.path}")
        return
    n_write = 0
    t0 = time.perf_counter()
    while not stop_evt.is_set():
        try:
            item = frame_q.get(timeout=0.2)
        except:
            continue
        if item is None:
            break
        try:
            writer.write(item)
            n_write += 1
        except Exception as e:
            print(f"[saver] write error: {e}")
    writer.release()
    elapsed = time.perf_counter() - t0
    if elapsed > 0:
        print(f"[saver] wrote {n_write} frames -> {cfg.path}  ({n_write/elapsed:.2f} fps)")

class AsyncVideoSaver:
    def __init__(self, cfg: SaverConfig):
        self.cfg = cfg
        self.stop_evt = Event()
        self.frame_q = Queue(maxsize=cfg.max_queue)
        self.proc = Process(target=_saver_proc, args=(self.frame_q, self.stop_evt, cfg), daemon=True)
        self.proc.start()

    def put(self, frame_bgr):
        try:
            self.frame_q.put_nowait(frame_bgr)
        except:
            if self.cfg.drop_old_when_full:
                try:
                    _ = self.frame_q.get_nowait()
                except:
                    pass
                try:
                    self.frame_q.put_nowait(frame_bgr)
                except:
                    pass
            else:
                self.frame_q.put(frame_bgr)

    def close(self):
        self.stop_evt.set()
        try:
            self.frame_q.put_nowait(None)
        except:
            pass
        self.proc.join(timeout=2.0)
