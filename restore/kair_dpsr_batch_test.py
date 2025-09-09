# restore/kair_dpsr_batch_test.py

import os
import sys
import cv2
import torch
import numpy as np
from importlib import import_module

class DPSRRestorer_Batch:
    def __init__(self, kair_repo: str, weights_path: str, scale: int = 2, use_cuda: bool = True):
        if not os.path.isdir(kair_repo):
            raise FileNotFoundError(f"KAIR repo not found at path: {kair_repo}")
        if kair_repo not in sys.path:
            sys.path.insert(0, kair_repo)

        self.torch = torch
        self.scale = scale
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"[DPSRRestorer_Batch] Using device: {self.device}")

        netmod = import_module("models.network_dpsr")
        MSRResNet_prior = getattr(netmod, "MSRResNet_prior")

        self.model = MSRResNet_prior(
            in_nc=4, out_nc=3, nc=96, nb=16,
            upscale=scale, act_mode='R', upsample_mode='pixelshuffle'
        )

        print(f"[DPSRRestorer_Batch] Loading weights from: {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        print("[DPSRRestorer_Batch] Model ready.")

    def __call__(self, bgr_batch: np.ndarray, noise_level: float = 0.0) -> np.ndarray:
        batch_size = bgr_batch.shape[0]
        rgb_batch_float = bgr_batch[:, :, :, ::-1].astype(np.float32) / 255.0
        rgb_tensor_batch = self.torch.from_numpy(rgb_batch_float.transpose(0, 3, 1, 2)).to(self.device)

        _, _, h, w = rgb_tensor_batch.size()
        noise_map_batch = self.torch.full(
            (batch_size, 1, h, w), noise_level / 255.0,
            dtype=rgb_tensor_batch.dtype, device=self.device
        )

        model_input_batch = self.torch.cat([rgb_tensor_batch, noise_map_batch], dim=1)

        with self.torch.no_grad():
            sr_tensor_batch = self.model(model_input_batch)

        sr_rgb_batch_np = sr_tensor_batch.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
        sr_bgr_batch_uint8 = (sr_rgb_batch_np * 255.0 + 0.5).astype(np.uint8)
        
        return sr_bgr_batch_uint8[:, :, :, ::-1]