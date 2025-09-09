# restore/pipeline_batch_test.py

import cv2
import numpy as np
from typing import Optional

from restore.kair_dncnn_batch_test import DnCNNRestorer_Batch
from restore.kair_dpsr_batch_test import DPSRRestorer_Batch

class Pipeline_Batch:
    def __init__(self, mode: str, dncnn: Optional[DnCNNRestorer_Batch], dpsr: Optional[DPSRRestorer_Batch], dpsr_noise: int = 0):
        self.mode = (mode or "dncnn+dpsr").lower()
        self.dncnn = dncnn
        self.dpsr = dpsr
        self.dpsr_noise = dpsr_noise

    def run(self, image_batch: np.ndarray) -> np.ndarray:
        try:
            h, w = image_batch.shape[1:3]

            if self.mode == "dncnn_only":
                if self.dncnn: return self.dncnn(image_batch)
            
            elif self.mode == "dpsr_only":
                if self.dpsr:
                    print("only dpsr")
                    sr_batch = self.dpsr(image_batch, noise_level=self.dpsr_noise)
                    return sr_batch
                return image_batch

            else:
                if self.dncnn and self.dpsr:
                    #print("im in the point!!!!")#------------------모드 인식은 됨
                    denoised_batch = self.dncnn(image_batch)
                    #print("complete denoiser")#-----------디노이저는 정상작동
                    sr_batch = self.dpsr(denoised_batch, noise_level=self.dpsr_noise)
                    print("COMPLETE??")#------------------------여기까지 안옴2---------------> dpsr class에 문제가 있는 듯
                    return sr_batch 
        except Exception as e:
            
            return image_batch
                


def build_pipeline_batch(mode: str, kair_cfg: dict) -> Pipeline_Batch:
    dncnn_batch = dpsr_batch = None
    

    try:
        dncnn_batch = DnCNNRestorer_Batch(kair_cfg["repo"], kair_cfg["dncnn_weights"])
    except Exception as e:
        print(f"[pipeline_batch] DnCNNRestorer_Batch init failed -> {e}")


    try:
        dpsr_batch = DPSRRestorer_Batch(
            kair_cfg["repo"], kair_cfg["dpsr_weights"],
            scale=int(kair_cfg.get("dpsr_scale", 2))
        )
    except Exception as e:
        print(f"[pipeline_batch] DPSRRestorer_Batch init failed -> {e}")

    return Pipeline_Batch(
        mode, dncnn_batch, dpsr_batch, 
        dpsr_noise=int(kair_cfg.get("dpsr_noise_level", 0))
    )
