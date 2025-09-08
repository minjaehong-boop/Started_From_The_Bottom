# core/assembler.py

import threading
from collections import defaultdict
import numpy as np
from utils.types import Box, TileResult

# 2D 가중치 마스크(블렌딩용)를 생성하는 함수
def _create_blend_mask(width, height, overlap):
    """타일 하나에 대한 2D 리니어 블렌딩 마스크를 생성합니다."""
    mask = np.ones((height, width), dtype=np.float32)
    
    if overlap > 0:
        # Horizontal gradient
        for i in range(overlap):
            mask[:, i] *= (i + 1) / (overlap + 1)
            mask[:, -i-1] *= (i + 1) / (overlap + 1)
        # Vertical gradient
        for i in range(overlap):
            mask[i, :] *= (i + 1) / (overlap + 1)
            mask[-i-1, :] *= (i + 1) / (overlap + 1)
    
    return mask

class FrameAssembler:
    """Gather all tiles for a given frame_id and assemble a single ROI patch with blending."""
    def __init__(self, roi: Box, rows: int, cols: int, overlap: int, max_lag: int = 120):
        self.roi = roi
        self.rows, self.cols = rows, cols
        self.tiles_per_frame = rows * cols
        self.overlap = overlap
        
        # 블렌딩을 위한 부동소수점 캔버스와 가중치 캔버스
        self.canvas_float = np.zeros((roi.h, roi.w, 3), dtype=np.float32)
        self.weights_canvas = np.zeros((roi.h, roi.w, 1), dtype=np.float32)
        
        # 각 타일 위치에 대한 블렌딩 마스크를 미리 계산
        self.precomputed_masks = self._precompute_masks(roi, rows, cols, overlap)

        self.cache = defaultdict(lambda: {
            'tiles': {},
            'accumulator': np.zeros_like(self.canvas_float),
            'weights': np.zeros_like(self.weights_canvas)
        })
        self.max_lag = max_lag
        self._last_gc = -1
        self._lock = threading.Lock()

    def _precompute_masks(self, roi, rows, cols, overlap):
        # 이 기능은 복잡성을 줄이기 위해 현재 구현에서는 제외하고,
        # 실시간으로 마스크를 생성하도록 합니다.
        return {}

    def push_and_try_assemble(self, tr: TileResult):
        out = None
        with self._lock:
            frame_cache = self.cache[tr.frame_id]
            frame_cache['tiles'][tr.tile_id] = tr

            # 1. Get the processed tile and its position
            processed_image = tr.image
            x, y, w, h = tr.box.x - self.roi.x, tr.box.y - self.roi.y, tr.box.w, tr.box.h
            
            # 2. Create blending mask for this tile
            blend_mask = _create_blend_mask(w, h, self.overlap)
            blend_mask = np.expand_dims(blend_mask, axis=2) # for broadcasting

            # 3. Add weighted pixel values to accumulator
            if processed_image.shape[:2] == (h, w):
                frame_cache['accumulator'][y:y+h, x:x+w] += processed_image.astype(np.float32) * blend_mask
                frame_cache['weights'][y:y+h, x:x+w] += blend_mask

            # GC: 너무 오래된 frame_id는 버림
            drop_before = tr.frame_id - self.max_lag
            if drop_before > self._last_gc:
                old = [fid for fid in self.cache.keys() if fid < drop_before]
                for fid in old:
                    del self.cache[fid]
                self._last_gc = drop_before

            if len(frame_cache['tiles']) == self.tiles_per_frame:
                print(f"Frame {tr.frame_id}: All tiles received, assembling with blending.")
                
                # 4. Normalize accumulator by weights to get final blended image
                # 0으로 나누는 것을 방지하기 위해 아주 작은 값을 더함
                final_canvas = frame_cache['accumulator'] / (frame_cache['weights'] + 1e-8)
                final_canvas = np.clip(final_canvas, 0, 255)
                
                del self.cache[tr.frame_id]
                out = (tr.frame_id, final_canvas.astype(np.uint8))
                
        return out