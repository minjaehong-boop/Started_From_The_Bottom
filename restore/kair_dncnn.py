import cv2, numpy as np, sys, os
import torch
from importlib import import_module
import time
class DnCNNRestorer:
    def __init__(self, kair_repo:str, weights_path:str, use_cuda:bool=True):
        print(f"[{time.time():.2f}] DnCNNRestorer 초기화 시작.")
        if not os.path.isdir(kair_repo):
            print(f"[{time.time():.2f}] 오류: KAIR repo 폴더를 찾을 수 없습니다: {kair_repo}", file=sys.stderr)
            raise FileNotFoundError(f"KAIR repo not found: {kair_repo}")
        
        if kair_repo not in sys.path:
            sys.path.insert(0, kair_repo)
            print(f"[{time.time():.2f}] KAIR repo 경로를 sys.path에 추가했습니다.")

        self.torch = torch
        self.device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
        print(f"[{time.time():.2f}] 사용할 장치: {self.device}")

        netmod = import_module("models.network_dncnn")
        print(f"[{time.time():.2f}] 'network_dncnn' 모듈 임포트 완료.")
        self.model = netmod.DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')

        try:
            print(f"[{time.time():.2f}] 모델 가중치 로딩 시작: {weights_path}")
            # torch>=2.4에서 추가된 weights_only=True 옵션 사용
            sd = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            # 이전 버전의 torch에 대한 대체 로직
            print(f"[{time.time():.2f}] 'weights_only' 옵션 없이 재시도...")
            sd = torch.load(weights_path, map_location=self.device)
        
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().to(self.device)
        for p in self.model.parameters(): 
            p.requires_grad_(False)
        self.torch.backends.cudnn.benchmark = True
        print(f"[{time.time():.2f}] DnCNNRestorer 초기화 완료.")

    def __call__(self, bgr:np.ndarray)->np.ndarray:
        print(f"[{time.time():.2f}] 추론 시작 - 입력 이미지 shape: {bgr.shape}")
        
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:,:,0].astype(np.float32)/255.0
        Cr = ycrcb[:,:,1]; Cb = ycrcb[:,:,2]
        print(f"[{time.time():.2f}] YCrCb 변환 및 Y채널 분리 완료. Y 채널 shape: {Y.shape}")
        
        with self.torch.no_grad():
            t = self.torch.from_numpy(Y).float().unsqueeze(0).unsqueeze(0).to(self.device)
            print(f"[{time.time():.2f}] NumPy -> Torch 텐서 변환 및 장치 이동 완료. 텐서 shape: {t.shape}")
            
            out = self.model(t).squeeze().clamp(0,1).detach().cpu().numpy()
            print(f"[{time.time():.2f}] 모델 추론 완료. 결과 텐서 shape: {out.shape}")
            
        Y8 = (out*255.0+0.5).astype(np.uint8)
        restored_bgr = cv2.cvtColor(np.stack([Y8,Cr,Cb],axis=2), cv2.COLOR_YCrCb2BGR)
        print(f"[{time.time():.2f}] YCrCb -> BGR 복원 완료. 최종 출력 shape: {restored_bgr.shape}")
        
        return restored_bgr