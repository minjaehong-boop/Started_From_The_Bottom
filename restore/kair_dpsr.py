# -*- coding: utf-8 -*-
import os
import sys
import cv2
import torch
import numpy as np
from importlib import import_module

class DPSRRestorer:
    """
    KAIR DPSR(MSRResNet_prior) wrapper.
    - Input: 3-channel BGR image + 1-channel noise level map = 4-channel concatenated tensor.
    - Output: Upscaled RGB image (x{scale}).
    """
    def __init__(self, kair_repo: str, weights_path: str, scale: int = 2, use_cuda: bool = True):
        # Ensure the KAIR repository is in the Python path to find the model definitions.
        # This check is now mainly for clarity, as the main script handles path injection.
        if not os.path.isdir(kair_repo):
            raise FileNotFoundError(f"KAIR repo not found at path: {kair_repo}")
        if kair_repo not in sys.path:
            sys.path.insert(0, kair_repo)

        self.torch = torch
        self.scale = scale
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"[DPSRRestorer] Using device: {self.device}")

        # Dynamically import the model network class from the KAIR repository
        netmod = import_module("models.network_dpsr")
        MSRResNet_prior = getattr(netmod, "MSRResNet_prior")

        # Initialize the model with the exact parameters from the reference script.
        # nc: number of channels, nb: number of residual blocks
        self.model = MSRResNet_prior(
            in_nc=4,           # 3 channels for RGB image + 1 for noise map
            out_nc=3,          # 3 channels for output RGB image
            nc=96,             # Number of feature channels
            nb=16,             # Number of residual blocks (conv layers)
            upscale=scale,
            act_mode='R',      # ReLU activation
            upsample_mode='pixelshuffle'
        )

        # Load the pre-trained model weights.
        # The official test script also uses strict=False, suggesting the state_dict
        # may contain keys not present in the model architecture (e.g., from older versions).
        print(f"[DPSRRestorer] Loading weights from: {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)

        # Set the model to evaluation mode and move it to the designated device.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        print("[DPSRRestorer] Model ready.")


    def __call__(self, bgr_image: np.ndarray, noise_level: float = 0.0) -> np.ndarray:
        # 1. Pre-process: Convert BGR NumPy array to RGB Torch tensor
        # The network expects RGB, float32, range [0, 1], and shape [N, C, H, W].
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 2. Create Noise Map
        # Create a 1-channel tensor with the same height and width as the input image,
        # filled with the normalized noise level.
        noise_map = torch.full(
            (1, 1, rgb_tensor.size(2), rgb_tensor.size(3)),
            noise_level / 255.0,
            dtype=rgb_tensor.dtype,
            device=self.device
        )

        # 3. Concatenate image and noise map to create the 4-channel input
        model_input = torch.cat([rgb_tensor, noise_map], dim=1)

        # 4. Run Inference
        # Execute the model within a no-gradient context for efficiency.
        with torch.no_grad():
            sr_tensor = self.model(model_input)

        # 5. Post-process: Convert output tensor back to BGR NumPy array
        # Clamp values to [0, 1], move to CPU, remove batch dimension, and convert layout.
        sr_rgb_np = sr_tensor.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        
        # Convert from float [0, 1] to uint8 [0, 255] and from RGB back to BGR.
        sr_bgr_image = (sr_rgb_np * 255.0 + 0.5).astype(np.uint8)
        sr_bgr_image = cv2.cvtColor(sr_bgr_image, cv2.COLOR_RGB2BGR)

        return sr_bgr_image