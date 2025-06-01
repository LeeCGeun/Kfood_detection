import time
import cv2
import numpy as np
from ultralytics import YOLO
from onnx_detect import preprocess, postprocess, session, input_name

IMG_PATH = "samgyeopsal.png"
REPEAT = 50

# ----------------------------
# YOLOv11 - Ultralytics
# ----------------------------
def test_yolo():
    model = YOLO("best.pt")
    times = []

    img = cv2.imread(IMG_PATH)
    for _ in range(REPEAT):
        start = time.time()
        _ = model(img)
        end = time.time()
        times.append(end - start)

    print(f"[YOLOv11] Avg inference time over {REPEAT} runs: {np.mean(times):.4f} sec")

# ----------------------------
# ONNX - onnxruntime
# ----------------------------
def test_onnx():
    times = []

    for _ in range(REPEAT):
        input_tensor, img, ori_w, ori_h = preprocess(IMG_PATH)
        start = time.time()
        outputs = session.run(None, {input_name: input_tensor})
        _ = postprocess(outputs, 0.2, ori_w, ori_h)
        end = time.time()
        times.append(end - start)

    print(f"[ONNX]   Avg inference time over {REPEAT} runs: {np.mean(times):.4f} sec")


if __name__ == "__main__":
    test_yolo()
    test_onnx()
