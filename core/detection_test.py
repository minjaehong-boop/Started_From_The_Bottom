# core/detection_test.py

import cv2
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    """YOLOv8 모델을 로드하고 객체 탐지를 수행하는 클래스"""

    def __init__(self, model_path: str, confidence_threshold: float = 0.4):
        """
        YOLOv8 모델을 초기화합니다.

        Args:
            model_path (str): YOLOv8 모델 가중치 파일 경로 (.pt 파일)
            confidence_threshold (float): 탐지를 위한 최소 신뢰도 값
        """
        try:
            self.model = YOLO(model_path)
            self.confidence_threshold = confidence_threshold
            print(f"YOLOv8 model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray):
        """
        입력된 프레임에서 객체를 탐지합니다.

        Args:
            frame (np.ndarray): 탐지를 수행할 이미지 프레임 (BGR)

        Returns:
            list: 탐지된 객체 정보 (바운딩 박스, 신뢰도, 클래스 ID) 리스트
        """
        if self.model is None:
            return []

        # YOLO 모델은 RGB 이미지를 기대하므로 BGR을 RGB로 변환하여 전달
        results = self.model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=self.confidence_threshold, verbose=False)
        
        detections = []
        # results[0].boxes থেকে 정보 추출
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append(((x1, y1, x2, y2), conf, cls_id))
            
        return detections

    def draw_detections(self, frame: np.ndarray, detections: list, label: str, color: tuple):
        """
        프레임에 탐지된 객체의 바운딩 박스와 레이블을 그립니다.

        Args:
            frame (np.ndarray): 시각화를 수행할 프레임
            detections (list): detect 메소드에서 반환된 탐지 정보 리스트
            label (str): 바운딩 박스 위에 표시할 접두사 (예: "Original")
            color (tuple): 바운딩 박스의 색상 (B, G, R)
        """
        for (x1, y1, x2, y2), conf, cls_id in detections:
            class_name = self.model.names[cls_id]
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 레이블 텍스트 생성
            text = f"{label}: {class_name} {conf:.2f}"
            
            # 텍스트 배경 및 텍스트 그리기
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return frame