# scripts/run_video_test.py

import os
import sys
import time
import argparse
import yaml
import cv2
import numpy as np
from multiprocessing import Queue, Event, set_start_method

# --- 프로젝트 모듈 임포트 ---
from utils.types import Box, TileResult
from core.tiling import split_grid
from core.assembler import FrameAssembler
from core.preview import LivePreview
from core.profiling import Profilers, _summ, format_hud
from core.io import VideoReader, FpsPacer
from external.async_saver import AsyncVideoSaver, SaverConfig
from restore.pipeline import build_pipeline
from restore.workers import spawn_workers
from core.detection_test import YoloDetector

def main(cfg_path: str):
    # 1. 설정 파일 로드
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. KAIR 및 YOLO 설정
    kair_repo_path = cfg.get("kair", {}).get("repo")
    if kair_repo_path and os.path.isdir(kair_repo_path) and kair_repo_path not in sys.path:
        sys.path.insert(0, kair_repo_path)

    yolo_cfg = cfg.get("yolo", {})
    yolo_enabled = yolo_cfg.get("enabled", False)

    # 3. 비디오 입출력 초기화
    vr = VideoReader(cfg["input"])
    info = vr.info()
    output_size_wh = (info.size_wh[0] * 2, info.size_wh[1])

    pacer = FpsPacer(info.fps) if cfg.get("lock_to_fps", True) else None
    preview = LivePreview("YOLOv8 Performance Test (Side-by-Side)", port=int(cfg.get("preview", {}).get("http_port", 8090)))
    saver = AsyncVideoSaver(SaverConfig(
        path=cfg["output"], fps=info.fps, size_wh=output_size_wh,
        fourcc=cfg.get("saver", {}).get("fourcc", "mp4v")
    ))

    # 4. ROI 및 타일 그리드 설정
    rows, cols = cfg.get("grid", [3, 3])
    tile_overlap = int(cfg.get("tile_overlap", 0))
    if cfg.get("roi") == "full":
        roi = Box(0, 0, info.size_wh[0], info.size_wh[1])
    else:
        x, y, w, h = cfg["roi"]
        roi = Box(x, y, w, h)
    tiles_defs = split_grid(roi, rows, cols, tile_overlap)

    # 5. 어셈블러, 워커, YOLO 탐지기 초기화
    assembler = FrameAssembler(roi, rows, cols, tile_overlap, max_lag=int(info.fps * 4))
    crop_q, done_q, stop_evt = Queue(maxsize=4096), Queue(maxsize=4096), Event()
    workers = spawn_workers(int(cfg.get("workers", 1)), crop_q, done_q, stop_evt, cfg["pipeline_mode"], cfg["kair"])

    detector = None
    if yolo_enabled:
        detector = YoloDetector(
            model_path=yolo_cfg["model_path"],
            confidence_threshold=float(yolo_cfg["confidence_threshold"])
        )

    # 6. 메인 루프 및 통계 변수
    prof = Profilers()
    frame_idx, processed_total = 0, 0
    latest_roi_patch = None
    last_read_frame = None
    total_detections_original = 0
    total_detections_enhanced = 0

    # ==================================
    # 7. 메인 루프
    # ==================================
    while True:
        ret, frame = vr.read()
        if not ret: break
        
        last_read_frame = frame.copy()

        if (frame_idx % int(cfg.get("frame_step", 1)) == 0):
            t_enq = time.perf_counter()
            for tid, (input_box, output_box) in enumerate(tiles_defs):
                b = input_box
                crop = frame[b.y:b.y+b.h, b.x:b.x+b.w].copy()
                try: crop_q.put_nowait((frame_idx, tid, b, crop, t_enq))
                except: pass

        while True:
            try:
                src_idx, tile_id, box, proc_tile, backend, t_enq, t_start, t_end = done_q.get_nowait()
                if proc_tile is not None and (proc_tile.shape[1], proc_tile.shape[0]) != (box.w, box.h):
                    proc_tile = cv2.resize(proc_tile, (box.w, box.h), interpolation=cv2.INTER_CUBIC)
                
                processed_total += 1
                pt = max(0.0, t_end - t_start); prof.last_ms = pt * 1000.0
                out = assembler.push_and_try_assemble(TileResult(src_idx, tile_id, box, proc_tile))
                if out is not None:
                    _, roi_patch = out
                    latest_roi_patch = roi_patch
            except: 
                break

        original_display = frame.copy()
        enhanced_display = frame.copy()
        
        if latest_roi_patch is not None:
            enhanced_display[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w] = latest_roi_patch

        detections_original, detections_enhanced = [], []
        if detector:
            detections_original = detector.detect(original_display)
            detections_enhanced = detector.detect(enhanced_display)
            total_detections_original += len(detections_original)
            total_detections_enhanced += len(detections_enhanced)
            
            detector.draw_detections(original_display, detections_original, "Orig", (0, 0, 255))
            detector.draw_detections(enhanced_display, detections_enhanced, "SR", (0, 255, 0))

        combined_frame = np.hstack((original_display, enhanced_display))
        hud_yolo = f"Detections -> Original: {len(detections_original)} | Enhanced: {len(detections_enhanced)}"
        cv2.putText(combined_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined_frame, "Enhanced", (10 + info.size_wh[0], 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame, hud_yolo, (10, info.size_wh[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        preview.show(combined_frame, waitkey_ms=1)
        saver.put(combined_frame)
        if pacer: pacer.sleep_for_frame(frame_idx)
        frame_idx += 1
        
    # ==================================
    # 8. Drain 루프 (수정된 로직)
    # ==================================
    print("\n[INFO] End of stream. Draining remaining tiles...")
    if last_read_frame is not None:
        # 마지막 원본 프레임을 왼쪽에 고정
        original_frozen_frame = last_read_frame.copy()
        if detector: # 마지막 원본 프레임의 탐지 결과도 고정
             detections_original_frozen = detector.detect(original_frozen_frame)
             detector.draw_detections(original_frozen_frame, detections_original_frozen, "Orig (Final)", (0, 0, 255))

        # Enhanced 프레임은 계속 업데이트될 것이므로 마지막 상태를 복사해 둠
        enhanced_last_known_frame = enhanced_display.copy() if 'enhanced_display' in locals() else last_read_frame.copy()

        DRAIN_IDLE_TIMEOUT_SEC = 3.0 # 3초 동안 새 타일이 없으면 종료
        last_tile_received_time = time.perf_counter()

        while time.perf_counter() - last_tile_received_time < DRAIN_IDLE_TIMEOUT_SEC:
            try:
                src_idx, tile_id, box, proc_tile, backend, t_enq, t_start, t_end = done_q.get(timeout=0.1)
                
                # 새 타일을 받았으므로 타이머 리셋
                last_tile_received_time = time.perf_counter()

                if proc_tile is not None and (proc_tile.shape[1], proc_tile.shape[0]) != (box.w, box.h):
                    proc_tile = cv2.resize(proc_tile, (box.w, box.h), interpolation=cv2.INTER_CUBIC)

                out = assembler.push_and_try_assemble(TileResult(src_idx, tile_id, box, proc_tile))
                if out is not None:
                    frame_id, roi_patch = out
                    print(f"[DRAIN] Assembled final frame {frame_id}...")
                    
                    # Enhanced 프레임의 ROI를 새 패치로 업데이트
                    enhanced_last_known_frame[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w] = roi_patch

                    # 업데이트된 Enhanced 프레임에서 YOLO 재탐지 및 통계 누적
                    if detector:
                        detections_enhanced = detector.detect(enhanced_last_known_frame)
                        total_detections_enhanced += len(detections_enhanced) # 여기서만 통계 누적
                        detector.draw_detections(enhanced_last_known_frame, detections_enhanced, "SR (Drain)", (0, 255, 0))
                    
                    # 좌우 합치기
                    combined_drain_frame = np.hstack((original_frozen_frame, enhanced_last_known_frame))
                    hud_drain = f"Detections -> Original: {len(detections_original_frozen)} | Enhanced: {len(detections_enhanced)}"
                    cv2.putText(combined_drain_frame, hud_drain, (10, info.size_wh[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    preview.show(combined_drain_frame, waitkey_ms=1)
                    saver.put(combined_drain_frame)
            except:
                # 큐가 비어 타임아웃 발생. 유휴 시간은 while 조건문이 체크함.
                pass
    print("[INFO] Drain complete.")
    
    # ==================================
    # 9. 종료 및 최종 성능 보고
    # ==================================
    print("\n[INFO] Shutting down...")
    stop_evt.set()
    saver.close()
    preview.close()
    vr.release()
    for p in workers: p.join(timeout=2.0)

    print("\n" + "="*50)
    print("      Quantitative Performance Analysis      ")
    print("="*50)
    print(f"Total Video Frames Processed: {frame_idx}")
    print(f"Total Tiles Processed by Workers: {processed_total}")
    print("-" * 50)
    print("YOLOv8 Object Detection Results:")
    print(f"  - Total Detections in Original Video: {total_detections_original}")
    print(f"  - Total Detections in Enhanced Video: {total_detections_enhanced}")
    
    avg_orig = total_detections_original / frame_idx if frame_idx > 0 else 0
    avg_enh = total_detections_enhanced / frame_idx if frame_idx > 0 else 0
    print(f"  - Avg Detections per Frame (Original): {avg_orig:.2f}")
    print(f"  - Avg Detections per Frame (Enhanced): {avg_enh:.2f}")
    
    improvement = ((total_detections_enhanced - total_detections_original) / total_detections_original * 100) if total_detections_original > 0 else (100.0 if total_detections_enhanced > 0 else 0.0)
    
    print("-" * 50)
    print(f"Overall Detection Improvement: {improvement:+.2f}%")
    print("="*50)

if __name__ == "__main__":
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description="Side-by-Side Video Enhancement and YOLOv8 Detection Pipeline.")
    parser.add_argument("--config", default="configs/default_test.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)