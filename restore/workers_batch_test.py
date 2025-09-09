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
            # 큐가 비어있으면 잠시 쉬어 CPU 사용률을 낮춤
            time.sleep(0.001)

        time_since_last = time.perf_counter() - last_batch_time
        
        # ===== 배치 처리 조건 수정 =====
        # 1. 배치가 꽉 찼거나
        # 2. 타임아웃이 지났고 버퍼에 아이템이 있으면서, 입력 큐가 비어있어 더 기다릴 필요가 없을 때
        process_now = len(batch_buffer) >= batch_size or \
                      (len(batch_buffer) > 0 and time_since_last > batch_timeout and crop_q.empty())

        if process_now:
            
            # =============================================================
            # ★★★★★ 에러 해결을 위한 핵심 수정 부분 ★★★★★
            # 1. 기준 크기를 첫 번째 타일로 설정
            first_item_shape = batch_buffer[0][3].shape
            target_h, target_w = first_item_shape[0], first_item_shape[1]

            images_resized = []
            for item in batch_buffer:
                img = item[3]
                # 2. 타일 크기가 기준과 다르면 리사이즈
                if img.shape[0] != target_h or img.shape[1] != target_w:
                    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                images_resized.append(img)
            # =============================================================

            # 3. 리사이즈된 이미지들로 안전하게 배치 생성
            images = np.stack(images_resized)
            
            t_start = time.perf_counter()
            processed_images = pipeline.run(images)
            t_end = time.perf_counter()
            
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