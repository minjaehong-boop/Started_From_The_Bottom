# restore/workers.py

import time, cv2
from multiprocessing import Process, Event, Queue
from typing import List, Tuple
from utils.types import TileTask, TileResult
from restore.pipeline import build_pipeline

def _worker(crop_q, done_q, stop_evt, wid, mode: str, kair_cfg: dict):
    print(f"[{time.time():.2f}] [Worker-{wid}] 워커 프로세스 시작.")
    
    try:
        pipeline = build_pipeline(mode, kair_cfg)
        print(f"[{time.time():.2f}] [Worker-{wid}] 파이프라인 생성 완료: {mode}")
    except Exception as e:
        print(f"[{time.time():.2f}] [Worker-{wid}] 파이프라인 생성 실패. 이유: {e}")
        pipeline = None
    
    # 파이프라인 모드에 따라 backend_name 설정
    if pipeline:
        if "dncnn" in mode and "dpsr" in mode:
            backend_name = "DnCNN+DPSR"
        elif "dncnn" in mode:
            backend_name = "DnCNN"
        elif "dpsr" in mode:
            backend_name = "DPSR"
        else:
            backend_name = "NLM" # Fallback
    else:
        backend_name = "NLM"

    while not stop_evt.is_set():
        try:
            item = crop_q.get(timeout=0.2)
        except:
            continue
        
        if item is None:
            print(f"[{time.time():.2f}] [Worker-{wid}] 종료 신호 수신, 루프 종료.")
            break
        
        frame_id, tile_id, box, image, t_enq = item
        t_start = time.perf_counter()
        
        print(f"[{time.time():.2f}] [Worker-{wid}] 프레임 {frame_id}의 타일 {tile_id} 처리 시작.")
        
        try:
            if pipeline:
                out = pipeline.run(image)
                print(f"[{time.time():.2f}] [Worker-{wid}] 프레임 {frame_id} 타일 {tile_id} 처리 완료.")
            else:
                out = image
                print(f"[{time.time():.2f}] [Worker-{wid}] 파이프라인 없음. 원본 이미지 반환.")

        except Exception as e:
            print(f"[{time.time():.2f}] [Worker-{wid}] 처리 중 예외 발생! 이유: {e}")
            out = image
            
        t_end = time.perf_counter()
        print(f"[{time.time():.2f}] [Worker-{wid}] 결과 타일 shape: {out.shape}, dtype: {out.dtype}")
        # `backend_name`을 추가하여 put 호출
        done_q.put((frame_id, tile_id, box, out, backend_name, t_enq, t_start, t_end))

def spawn_workers(num, crop_q, done_q, stop_evt, mode: str, kair_cfg: dict):
    ps = []
    print(f"[{time.time():.2f}] 워커 프로세스 {num}개 생성 시작.")
    for i in range(max(1, num)):
        p = Process(target=_worker, args=(crop_q, done_q, stop_evt, i, mode, kair_cfg), daemon=True)
        p.start()
        ps.append(p)
        print(f"[{time.time():.2f}] [Worker-{i}] 생성 완료.")
    return ps