# restore/workers_batch_test.py

import time
import cv2
import numpy as np
from multiprocessing import Process, Event, Queue
from restore.pipeline_batch_test import build_pipeline_batch

def _worker_batch(crop_q: Queue, done_q: Queue, stop_evt: Event, wid: int, mode: str, kair_cfg: dict, batch_size: int, batch_timeout: float):
    print(f"[{time.time():.2f}] [Worker-{wid}] 배치 워커 프로세스 시작 (Batch Size: {batch_size}).")
    pipeline = build_pipeline_batch(mode, kair_cfg)
    
    batch_buffer = []
    last_batch_time = time.perf_counter()

    while not stop_evt.is_set():
        try:
            item = crop_q.get_nowait()
            batch_buffer.append(item)
        except:
            pass

        time_since_last = time.perf_counter() - last_batch_time

        process_now = len(batch_buffer) >= batch_size or \
                      (len(batch_buffer) > 0 and time_since_last > batch_timeout and crop_q.empty())

        if process_now:
            first_item_shape = batch_buffer[0][3].shape
            target_h, target_w = first_item_shape[0], first_item_shape[1]

            images_resized = []
            for item in batch_buffer:
                img = item[3] #32회 돌아감
                images_resized.append(img)

            #배치 생성
            images = np.stack(images_resized)
            
            t_start = time.perf_counter()
            processed_images = pipeline.run(images)
            t_end = time.perf_counter()
            print(" im hererererere")#-----------------------여기까지 안옴 파이프라인으로 들어가서 확인------------
            
            print(f"[{time.time():.2f}] [Worker-{wid}] {len(batch_buffer)}개 타일 배치 처리 완료 ({(t_end - t_start)*1000:.2f}ms)")

            for i, original_item in enumerate(batch_buffer):
                frame_id, tile_id, box, _, t_enq = original_item
                processed_tile = processed_images[i]
                done_q.put((frame_id, tile_id, box, processed_tile, "BATCH_PIPELINE", t_enq, t_start, t_end))

            batch_buffer = []
            last_batch_time = time.perf_counter()

def spawn_workers_batch(num, crop_q, done_q, stop_evt, mode, kair_cfg, batch_size, batch_timeout):
    ps = []
    # 배치 처리는 단일 워커가 GPU를 독점하는 것이 효율적
    for i in range(max(1, num)):
        p = Process(target=_worker_batch, args=(crop_q, done_q, stop_evt, i, mode, kair_cfg, batch_size, batch_timeout), daemon=True)
        p.start()
        ps.append(p)
    return ps
