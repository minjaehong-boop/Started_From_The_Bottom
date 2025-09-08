import cv2
from typing import Optional
from restore.kair_dncnn import DnCNNRestorer
from restore.kair_dpsr import DPSRRestorer

class Pipeline:
    def __init__(self, mode:str, dncnn: Optional[DnCNNRestorer], dpsr: Optional[DPSRRestorer], dpsr_noise:int=0):
        self.mode = (mode or "dncnn+dpsr").lower()
        self.dncnn = dncnn
        self.dpsr = dpsr
        self.dpsr_noise = dpsr_noise

    def run(self, img):
        try:
            if self.mode == "dncnn_only":
                if self.dncnn: return self.dncnn(img)
                return cv2.fastNlMeansDenoisingColored(img, None, 3,3,7,21)
            elif self.mode == "dpsr_only":
                if self.dpsr:
                    sr = self.dpsr(img, noise_level=self.dpsr_noise)
                    return sr # cv2.resize 제거
                return img
            else:
                if self.dncnn and self.dpsr:
                    den = self.dncnn(img)
                    sr  = self.dpsr(den, noise_level=self.dpsr_noise)
                    return sr # cv2.resize 제거
                return cv2.fastNlMeansDenoisingColored(img, None, 3,3,7,21)
        except Exception:
            return img

# restore/pipeline.py 수정 (디버깅용)

def build_pipeline(mode:str, kair_cfg:dict) -> Pipeline:
    dncnn = dpsr = None
    try:
        dncnn = DnCNNRestorer(kair_cfg["repo"], kair_cfg["dncnn_weights"], use_cuda=True)
        print("[pipeline] DnCNN ready")
    except Exception as e:
        print("[pipeline] DnCNN init failed ->", e)
        raise e  # 예외를 다시 발생시켜 프로그램을 중단시킴
    try:
        dpsr = DPSRRestorer(kair_cfg["repo"], kair_cfg["dpsr_weights"],
                            scale=int(kair_cfg.get("dpsr_scale",2)), use_cuda=True)
        print("[pipeline] DPSR ready")
    except Exception as e:
        print("[pipeline] DPSR init failed ->", e)
        raise e  # 예외를 다시 발생시켜 프로그램을 중단시킴
    return Pipeline(mode, dncnn, dpsr, dpsr_noise=int(kair_cfg.get("dpsr_noise_level", 0)))