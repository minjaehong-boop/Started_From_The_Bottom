import cv2
import numpy as np 

cap = cv2.VideoCapture("/home/alswoghd/deepet_ws/after_score/testsets/videos/original/cloud.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("/home/alswoghd/deepet_ws/after_score/testsets/videos/downscale/degraded_cloud.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))


while True:
    ret, frame = cap.read()
    if not ret:
        break
    

    small = cv2.resize(frame, (frame.shape[1]//8, frame.shape[0]//8))
    degraded = cv2.resize(small, (frame.shape[1], frame.shape[0]))

    sigma = 25  
    noise = np.random.normal(0, sigma, degraded.shape).astype(np.float32) 
    noisy_degraded = degraded.astype(np.float32) + noise                  
    noisy_degraded = np.clip(noisy_degraded, 0, 255)                      
    final_output = noisy_degraded.astype(np.uint8)                       

    out.write(final_output)

cap.release()
out.release()
cv2.destroyAllWindows()
print("작업 완료!")