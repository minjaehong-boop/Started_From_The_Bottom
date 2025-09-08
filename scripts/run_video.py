# scripts/run_video.py

import os
import sys
import time
import argparse
import yaml
import cv2
import numpy as np
from multiprocessing import Queue, Event
from utils.types import Box, TileResult
from core.tiling import split_grid
from core.assembler import FrameAssembler
from core.preview import LivePreview
from core.profiling import Profilers, _summ, format_hud
from core.io import VideoReader, FpsPacer
from external.async_saver import AsyncVideoSaver, SaverConfig
from restore.pipeline import build_pipeline
from restore.workers import spawn_workers
import multiprocessing as mp

def main(cfg_path: str):
    # 1. 설정 파일 로드
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. KAIR 경로 설정
    kair_repo_path = cfg.get("kair", {}).get("repo")
    if kair_repo_path and os.path.isdir(kair_repo_path) and kair_repo_path not in sys.path:
        sys.path.insert(0, kair_repo_path)
        print(f"KAIR repository path added to sys.path: {kair_repo_path}")

    # 3. 비디오 입출력 및 미리보기 객체 초기화
    vr = VideoReader(cfg["input"])
    info = vr.info()
    pacer = FpsPacer(info.fps) if cfg.get("lock_to_fps", True) else None
    preview = LivePreview("preview (mod)", port=int(cfg.get("preview", {}).get("http_port", 8090)))
    
    output_path = cfg.get("output", "output.mp4")
    saver = AsyncVideoSaver(SaverConfig(
        path=output_path, 
        fps=info.fps, 
        size_wh=info.size_wh,
        fourcc=cfg.get("saver", {}).get("fourcc", "mp4v"),
        max_queue=int(cfg.get("saver", {}).get("max_queue", 4096)),
        drop_old_when_full=bool(cfg.get("saver", {}).get("drop_old_when_full", False))
    ))
    print(f"Output will be saved to: {output_path}")

    # 4. ROI 및 타일 그리드 설정
    rows, cols = cfg.get("grid", [3, 3])
    tile_overlap = int(cfg.get("tile_overlap", 0))

    if cfg.get("roi") == "full":
        roi = Box(0, 0, info.size_wh[0], info.size_wh[1])
        print(f"Using full frame as ROI: {info.size_wh[0]}x{info.size_wh[1]}")
    else:
        x, y, w, h = cfg["roi"]
        roi = Box(x, y, w, h)
        print(f"Using specific ROI: x={x}, y={y}, w={w}, h={h}")

    tiles_defs = split_grid(roi, rows, cols, tile_overlap)

    # 5. 어셈블러 및 워커 프로세스 생성
    lag_cfg = int(cfg.get("assembler", {}).get("max_lag_frames", int(info.fps * 4)))
    assembler = FrameAssembler(roi, rows, cols, tile_overlap, max_lag=lag_cfg)
    
    mode = cfg["pipeline_mode"]
    kair_cfg = cfg["kair"]
    num_workers = int(cfg.get("workers", 1))
    crop_q, done_q, stop_evt = Queue(maxsize=4096), Queue(maxsize=4096), Event()
    
    print(f"Spawning {num_workers} worker(s)...")
    workers = spawn_workers(num_workers, crop_q, done_q, stop_evt, mode, kair_cfg)

    # 6. 메인 루프 변수 초기화
    prof = Profilers()
    processed_total = 0
    dncnn_count = 0
    latest_roi_patch = None
    last_update_frame = -1
    last_read_frame = None
    frame_idx = 0
    FRAME_STEP = int(cfg.get("frame_step", 1))

    # ==================================
    # 7. 캡처 및 처리 루프
    # ==================================
    while True:
        ret, frame = vr.read()
        if not ret:
            print("\nEnd of video stream.")
            break
        
        last_read_frame = frame.copy()

        # 프레임 스텝에 따라 타일 큐잉
        if (frame_idx % FRAME_STEP == 0):
            t_enq = time.perf_counter()
            for tid, (input_box, output_box) in enumerate(tiles_defs):
                b = input_box
                crop = frame[b.y:b.y+b.h, b.x:b.x+b.w].copy()
                try:
                    crop_q.put_nowait((frame_idx, tid, b, crop, t_enq))
                except:
                    pass # 큐가 가득 차면 타일 드롭

        # 결과 큐에서 처리된 타일 가져오기
        while True:
            try:
                src_idx, tile_id, box, proc_tile, backend, t_enq, t_start, t_end = done_q.get_nowait()
            except:
                break
            
            if proc_tile is None: continue
            
            # 후처리된 타일 크기가 원본과 다를 경우 리사이즈
            if (proc_tile.shape[1], proc_tile.shape[0]) != (box.w, box.h):
                proc_tile = cv2.resize(proc_tile, (box.w, box.h), interpolation=cv2.INTER_CUBIC)

            processed_total += 1
            if backend and "DnCNN" in backend:
                dncnn_count += 1
            
            pt = max(0.0, t_end - t_start)
            prof.last_ms = pt * 1000.0
            prof.proctime.append(pt)
            
            # 어셈블러로 타일 전달 및 완성된 ROI 패치 확인
            out = assembler.push_and_try_assemble(TileResult(src_idx, tile_id, box, proc_tile))
            if out is not None:
                _, roi_patch = out
                latest_roi_patch = roi_patch
                last_update_frame = frame_idx

        # 완성된 ROI 패치가 있으면 프레임에 적용
        applied = latest_roi_patch is not None
        if applied:
            frame[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w] = latest_roi_patch
            
        # HUD 정보 표시
        ratio = (dncnn_count / processed_total * 100.0) if processed_total else 0.0
        since = (frame_idx - last_update_frame) if last_update_frame >= 0 else -1
        hud = format_hud(FRAME_STEP, (rows, cols), processed_total, ratio, since, prof.last_ms)
        
        color = (0, 255, 0) if applied else (0, 0, 255)
        status = "Assembled" if applied else "Waiting for tiles..."
        cv2.rectangle(frame, (roi.x, roi.y), (roi.x+roi.w, roi.y+roi.h), color, 2)
        cv2.putText(frame, status, (roi.x, roi.y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 220, 10), 2, cv2.LINE_AA)

        # 미리보기, 저장, FPS 동기화
        if bool(cfg.get("preview", {}).get("show_window", True)):
            preview.show(frame, waitkey_ms=1)
        saver.put(frame)
        if pacer:
            pacer.sleep_for_frame(frame_idx)
        
        frame_idx += 1

    # ==================================
    # 8. Drain 루프 (남은 타일 처리)
    # ==================================
    print("\nEntering drain mode to process remaining tiles...")
    
    # 마지막 프레임을 캔버스로 사용, 없으면 검은 화면 생성
    canvas = last_read_frame if last_read_frame is not None else np.zeros((info.size_wh[1], info.size_wh[0], 3), dtype=np.uint8)
    
    DRAIN_IDLE_TIMEOUT = 1.0  # drain 중 1초간 새 결과 없으면 종료
    last_drain_update_t = time.perf_counter()

    while time.perf_counter() - last_drain_update_t < DRAIN_IDLE_TIMEOUT:
        try:
            # 타임아웃을 짧게 주어 루프가 너무 오래 블로킹되지 않도록 함
            src_idx, tile_id, box, proc_tile, backend, t_enq, t_start, t_end = done_q.get(timeout=0.1)
        except:
            # 큐가 비어 타임아웃 발생 시, 유휴 시간 체크
            if done_q.empty():
                break
            continue

        last_drain_update_t = time.perf_counter()
        if proc_tile is None: continue

        if (proc_tile.shape[1], proc_tile.shape[0]) != (box.w, box.h):
            proc_tile = cv2.resize(proc_tile, (box.w, box.h), interpolation=cv2.INTER_CUBIC)

        out = assembler.push_and_try_assemble(TileResult(src_idx, tile_id, box, proc_tile))
        if out is not None:
            frame_id, roi_patch = out
            print(f"[DRAIN] Assembled and saving final frame {frame_id}...")
            
            # 완성된 ROI를 캔버스에 적용
            canvas[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w] = roi_patch
            
            # HUD 업데이트
            cv2.putText(canvas, f"[DRAIN] Assembled frame: {frame_id}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2, cv2.LINE_AA)
            
            # 미리보기 및 저장
            if bool(cfg.get("preview", {}).get("show_window", True)):
                preview.show(canvas, waitkey_ms=1)
            saver.put(canvas.copy())

    print("Drain mode finished.")

    # ==================================
    # 9. 종료 처리
    # ==================================
    print("Shutting down...")
    stop_evt.set()
    saver.close()
    preview.close()
    vr.release()
    
    # 워커 프로세스 종료 보장
    for p in workers:
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()

    print("\n[ 최종 프로파일링 정보 ]")
    print(_summ("ProcTime", prof.proctime))
    print(f"Total frames processed: {frame_idx}, Total tiles processed: {processed_total}")
    print("Shutdown complete.")

if __name__ == "__main__":
    # CUDA 사용 시 'spawn' 방식이 안정적일 수 있음
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Video enhancement pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    
    main(args.config)