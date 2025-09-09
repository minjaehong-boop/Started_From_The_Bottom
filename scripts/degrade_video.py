from __future__ import annotations
import cv2
import numpy as np
import argparse
import math
import os
import time
import random

def add_gaussian_noise(img, sigma=10.0):
    if sigma <= 0:
        return img
    h, w, c = img.shape
    noise = np.random.normal(0, sigma, (h, w, c)).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def down_up(img, scale=0.5, interp_down=cv2.INTER_AREA, interp_up=cv2.INTER_CUBIC):
    if scale >= 1.0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=interp_down)
    back  = cv2.resize(small, (w, h), interpolation=interp_up)
    return back

def gaussian_blur(img, ksize=5, sigma=0):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    if ksize <= 1:
        return img
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def motion_blur(img, ksize=9, angle=0.0):
    k = max(1, int(ksize))
    if k < 3:
        return img
    # PSF 생성(선형 커널)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k//2, :] = 1.0
    # 회전
    M = cv2.getRotationMatrix2D((k/2-0.5, k/2-0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel = kernel / (kernel.sum() + 1e-8)
    out = cv2.filter2D(img, -1, kernel)
    return out

def jpeg_compress_artifact(img, quality=30):
    # 프레임을 JPEG로 인코딩 후 디코딩 → 손실압축 아티팩트 유도
    quality = int(np.clip(quality, 5, 95))
    enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

def gamma_correction(img, gamma=1.0):
    if abs(gamma - 1.0) < 1e-3:
        return img
    inv = 1.0 / max(1e-6, gamma)
    table = (np.arange(256) / 255.0) ** inv
    table = np.clip(table * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)

def color_jitter(img, brightness=0.0, contrast=0.0, saturation=0.0):
    out = img.astype(np.float32)
    if brightness != 0.0:
        out += brightness
    if contrast != 0.0:
        out = (out - 127.5) * (1.0 + contrast) + 127.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    if saturation != 0.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1.0 + saturation), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_BGR2BGR)
    return out

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="demo.mp4")
    p.add_argument("--output", type=str, default="demo_degraded.mp4")
    p.add_argument("--preset", type=str, default="medium", choices=["mild","medium","heavy","custom"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show", action="store_true")

    # Custom controls (preset=custom 이거나 preset 덮어쓰기)
    p.add_argument("--downscale", type=float, default=None, help="0<scale<1, e.g., 0.5")
    p.add_argument("--blur", type=int, default=None, help="Gaussian blur ksize")
    p.add_argument("--motion", type=int, default=None, help="motion blur length (kernel size)")
    p.add_argument("--motion_angle", type=float, default=None, help="motion blur angle in degrees")
    p.add_argument("--noise_sigma", type=float, default=None, help="Gaussian noise sigma")
    p.add_argument("--jpeg_q", type=int, default=None, help="JPEG quality (5~95)")
    p.add_argument("--gamma", type=float, default=None, help="gamma > 0. e.g., 1.2 (darker), 0.8 (brighter)")
    p.add_argument("--brightness", type=float, default=None, help="additive shift [-50..50]")
    p.add_argument("--contrast", type=float, default=None, help="contrast scale delta e.g., 0.2 (=> *1.2)")
    p.add_argument("--saturation", type=float, default=None, help="saturation scale delta e.g., 0.2")

    return p.parse_args()

def resolve_preset(args):
    # 기본 preset 값
    if args.preset == "mild":
        cfg = dict(downscale=0.85, blur=3, motion=0, motion_angle=0, noise_sigma=4,  jpeg_q=70, gamma=1.0, brightness=0.0, contrast=0.0, saturation=0.0)
    elif args.preset == "medium":
        cfg = dict(downscale=0.6,  blur=5, motion=7, motion_angle=10, noise_sigma=10, jpeg_q=40, gamma=1.1, brightness=-5.0, contrast=0.1, saturation=0.0)
    elif args.preset == "heavy":
        cfg = dict(downscale=0.4,  blur=7, motion=11, motion_angle=25, noise_sigma=18, jpeg_q=20, gamma=1.2, brightness=-10., contrast=0.2, saturation=-0.1)
    else:  # custom
        cfg = dict(downscale=1,  blur=0, motion=0, motion_angle=0, noise_sigma=20, jpeg_q=70, gamma=1.0, brightness=0.0, contrast=0.0, saturation=0.0)

    # CLI 인자가 주어지면 덮어씀
    for k in list(cfg.keys()):
        v = getattr(args, k if k != "jpeg_q" else "jpeg_q")
        if v is not None:
            cfg[k] = v
    return cfg

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input}")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Cannot read first frame.")
    H, W = frame.shape[:2]

    cfg = resolve_preset(args)
    print("[Preset]", cfg)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vw = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    t0 = time.time()
    n = 0
    try:
        while True:
            if frame is None:
                break

            out = frame.copy()

            # 1) 해상도 저하 → 업샘플
            if cfg["downscale"] and cfg["downscale"] < 1.0:
                out = down_up(out, cfg["downscale"], interp_down=cv2.INTER_AREA, interp_up=cv2.INTER_CUBIC)

            # 2) 가우시안 블러
            if cfg["blur"] and int(cfg["blur"]) > 1:
                out = gaussian_blur(out, cfg["blur"], sigma=0)

            # 3) 모션 블러
            if cfg["motion"] and int(cfg["motion"]) > 2:
                angle = float(cfg.get("motion_angle", 0.0) or 0.0)
                out = motion_blur(out, cfg["motion"], angle)

            # 4) JPEG 아티팩트
            if cfg["jpeg_q"]:
                out = jpeg_compress_artifact(out, cfg["jpeg_q"])

            # 5) 가우시안 노이즈
            if cfg["noise_sigma"] and cfg["noise_sigma"] > 0:
                out = add_gaussian_noise(out, cfg["noise_sigma"])

            # 6) 감마/밝기/대비/채도
            if cfg["gamma"] and cfg["gamma"] > 0:
                out = gamma_correction(out, cfg["gamma"])
            out = color_jitter(out,
                               brightness=cfg.get("brightness", 0.0) or 0.0,
                               contrast=cfg.get("contrast", 0.0) or 0.0,
                               saturation=cfg.get("saturation", 0.0) or 0.0)

            vw.write(out)

            if args.show:
                disp = out.copy()
                cv2.putText(disp, f"frame:{n}", (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("degraded", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            ok, frame = cap.read()
            if not ok:
                break
            n += 1
    finally:
        cap.release()
        vw.release()
        cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f"[Done] {n} frames -> {args.output}  ({dt:.2f}s)")

if __name__ == "__main__":
    main()
