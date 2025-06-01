from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from datetime import datetime
from onnx_detect import run  # onnx_detect.py의 run 함수 사용

app = FastAPI()

# 디렉토리 생성
UPLOAD_DIR = "uploads"  # 클라이언트 업로드 이미지 저장 폴더
RESULT_DIR = "results"  # 결과 이미지 저장 폴더
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 파일 이름 및 경로 설정
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    result_path = os.path.join(RESULT_DIR, f"result_{filename}")

    # 이미지 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ONNX 추론 실행
    result = run(img_path=file_path, save_path=result_path, conf_thres=0.3)

    # 응답 반환 (JSON)
    return JSONResponse(content={
        "labels": result,
        "saved_image": result_path
    })
