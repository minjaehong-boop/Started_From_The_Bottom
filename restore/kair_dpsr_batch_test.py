# restore/kair_dpsr_batch_test.py

import os
import sys
import cv2
import torch
import numpy as np
from importlib import import_module
from typing import List

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
            in_nc=4, out_nc=3, nc=96, nb=16, upscale=scale,
            act_mode='R', upsample_mode='pixelshuffle'
        )

        print(f"[DPSRRestorer_Batch] Loading weights from: {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        print("[DPSRRestorer_Batch] Model ready.")

    def __call__(self, bgr_images: List[np.ndarray], noise_level: float = 0.0) -> List[np.ndarray]:
        if not bgr_images:
            return [] #---------여기엔 일단 안걸림

        rgb_tensors = []
        for bgr_image in bgr_images:
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            rgb_tensor = torch.from_numpy(rgb_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            rgb_tensors.append(rgb_tensor)
        
        batch_rgb_tensor = torch.cat(rgb_tensors, dim=0)

        batch_size, _, h, w = batch_rgb_tensor.size()
        noise_map = torch.full(
            (batch_size, 1, h, w),
            noise_level / 255.0,
            dtype=batch_rgb_tensor.dtype,
            device=self.device
        )

        model_input = torch.cat([batch_rgb_tensor, noise_map], dim=1)

        with torch.no_grad():
            sr_tensor_batch = self.model(model_input)

        output_images = []
        for sr_tensor in sr_tensor_batch.split(1, dim=0):
            sr_rgb_np = sr_tensor.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

            sr_bgr_image = (sr_rgb_np * 255.0 + 0.5).astype(np.uint8)
            sr_bgr_image = cv2.cvtColor(sr_bgr_image, cv2.COLOR_RGB2BGR)
            output_images.append(sr_bgr_image)#리스트?

        return output_images
