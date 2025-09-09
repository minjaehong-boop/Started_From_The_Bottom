# restore/kair_dncnn_batch_test.py

import cv2
import numpy as np
import sys
import os
import torch
from importlib import import_module
import time

class DnCNNRestorer_Batch:
    def __init__(self, kair_repo:str, weights_path:str, use_cuda:bool=True):
        print(f"[{time.time():.2f}] DnCNNRestorer_Batch 초기화 시작.")
        if not os.path.isdir(kair_repo):
            raise FileNotFoundError(f"KAIR repo not found: {kair_repo}")
        
        if kair_repo not in sys.path:
            sys.path.insert(0, kair_repo)

        self.torch = torch
        self.device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
        print(f"[{time.time():.2f}] DnCNNRestorer_Batch 장치: {self.device}")

        netmod = import_module("models.network_dncnn")
        self.model = netmod.DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
        
        sd = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().to(self.device)
        print(f"[{time.time():.2f}] DnCNNRestorer_Batch 초기화 완료.")

    def __call__(self, bgr_batch: np.ndarray) -> np.ndarray:
        batch_size = bgr_batch.shape[0]
        
        # YCrCb 변환 및 Y 채널 분리 (배치)
        ycrcb_batch = [cv2.cvtColor(bgr_batch[i], cv2.COLOR_BGR2YCrCb) for i in range(batch_size)]
        y_planes = np.stack([img[:,:,0] for img in ycrcb_batch]).astype(np.float32) / 255.0
        cr_planes = [img[:,:,1] for img in ycrcb_batch]
        cb_planes = [img[:,:,2] for img in ycrcb_batch]

        with self.torch.no_grad():
            # (N, H, W) -> (N, 1, H, W) 배치 텐서 생성
            y_tensor_batch = self.torch.from_numpy(y_planes).float().unsqueeze(1).to(self.device)
            
            # 모델이 배치 전체를 한 번에 처리
            out_tensor_batch = self.model(y_tensor_batch)
            
            y_hat_batch = out_tensor_batch.squeeze(1).clamp(0, 1).cpu().numpy()

        # 결과 재조합
        restored_bgrs = []
        for i in range(batch_size):
            y8 = (y_hat_batch[i] * 255.0 + 0.5).astype(np.uint8)
            out_ycrcb = np.stack([y8, cr_planes[i], cb_planes[i]], axis=2)
            restored_bgrs.append(cv2.cvtColor(out_ycrcb, cv2.COLOR_YCrCb2BGR))
            
        return np.stack(restored_bgrs)