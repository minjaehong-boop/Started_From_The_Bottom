# core/tiling.py

from typing import List, Tuple
from utils.types import Box

def split_grid(roi: Box, rows: int, cols: int, overlap: int) -> List[Tuple[Box, Box]]:
    """
    ROI를 오버랩을 포함하는 그리드로 분할합니다.

    Args:
        roi: 분할할 전체 영역 (Region of Interest).
        rows: 그리드의 행 수.
        cols: 그리드의 열 수.
        overlap: 타일 간 겹침 픽셀 수.

    Returns:
        A list of tuples, where each tuple contains:
        - input_box: 오버랩이 포함된, 소스 프레임에서 잘라낼 더 큰 영역.
        - output_box: 오버랩이 없는, 최종 캔버스에 위치할 실제 타일 영역.
    """
    tiles = []
    base_w = roi.w // cols
    base_h = roi.h // rows
    
    # 짝수 오버랩을 보장하여 절반씩 나누기 쉽게 함
    overlap = overlap if overlap % 2 == 0 else overlap + 1
    half_overlap = overlap // 2

    for r in range(rows):
        for c in range(cols):
            # 1. Calculate the final output box (no overlap)
            out_x = roi.x + c * base_w
            out_y = roi.y + r * base_h
            # 마지막 행/열의 크기 보정
            out_w = base_w if c < cols - 1 else (roi.x + roi.w) - out_x
            out_h = base_h if r < rows - 1 else (roi.y + roi.h) - out_y
            output_box = Box(out_x, out_y, out_w, out_h)

            # 2. Calculate the input box by adding overlap padding
            in_x = out_x - half_overlap
            in_y = out_y - half_overlap
            in_w = out_w + overlap
            in_h = out_h + overlap

            # 3. Adjust for boundaries (don't go outside the main ROI)
            # 첫 행/열과 마지막 행/열의 패딩을 조절
            if c == 0: # First column
                in_x = out_x
                in_w = out_w + half_overlap
            if c == cols - 1: # Last column
                in_w = out_w + half_overlap
            if r == 0: # First row
                in_y = out_y
                in_h = out_h + half_overlap
            if r == rows - 1: # Last row
                in_h = out_h + half_overlap
            
            # 최종적으로 ROI 경계 내에 있도록 클리핑
            final_in_x = max(roi.x, in_x)
            final_in_y = max(roi.y, in_y)
            final_in_w = min(in_w, roi.w - (final_in_x - roi.x))
            final_in_h = min(in_h, roi.h - (final_in_y - roi.y))
            
            input_box = Box(final_in_x, final_in_y, final_in_w, final_in_h)
            
            tiles.append((input_box, output_box))
            
    return tiles