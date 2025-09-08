import statistics
from collections import deque

class Profilers:
    def __init__(self, cap:int=20000):
        self.queuewait = deque(maxlen=cap)
        self.proctime  = deque(maxlen=cap)
        self.applylag  = deque(maxlen=cap)
        self.rtt       = deque(maxlen=cap)
        self.last_ms   = -1.0

def _summ(name, data):
    if not data:
        return f"{name}: n=0"
    arr = list(data)
    avg = sum(arr) / len(arr)
    med = statistics.median(arr)
    p90 = statistics.quantiles(arr, n=10)[8] if len(arr) >= 10 else max(arr)
    return f"{name}: n={len(arr)}  avg={avg*1000:.2f}ms  med={med*1000:.2f}ms  p90={p90*1000:.2f}ms"

def format_hud(step, grid_rc, done, dncnn_ratio, frames_since_update, last_proc_ms):
    return (f"step={step}  tiles=({grid_rc[0]}x{grid_rc[1]})  done={done}  "
            f"DnCNN={dncnn_ratio:.1f}%  updated {frames_since_update}f ago  Proc={last_proc_ms:.1f}ms")
