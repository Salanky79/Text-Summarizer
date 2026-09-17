import os
import uvicorn
from enum import Enum
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI(title="Text Summarization API")

# 1. Khởi tạo Pipeline & Caching Model 1 lần duy nhất khi ứng dụng startup
try:
    prediction_pipeline = PredictionPipeline()
except Exception as e:
    prediction_pipeline = None
    print(f"Lỗi khởi tạo PredictionPipeline: {e}")


# 2. Định nghĩa danh sách các kiểu độ dài bằng Enum (tạo dropdown trên Swagger)
class SummaryType(str, Enum):
    short = "Short"
    medium = "Medium"
    long = "Long"
    detailed = "Detailed"


# 3. Pydantic Model để validate đầu vào API
class TextRequest(BaseModel):
    text: str
    summary_type: SummaryType = SummaryType.short  # Mặc định chọn Short


@app.get("/", tags=["authentication"])
async def index():
    """Điều hướng trang chủ về FastAPI Swagger UI Docs"""
    return RedirectResponse(url="/docs")


def run_training_pipeline():
    """Hàm chạy pipeline huấn luyện ngầm"""
    os.system("python main.py")


@app.get("/train", tags=["pipeline"])
async def training(background_tasks: BackgroundTasks):
    """
    Chạy pipeline huấn luyện ngầm dưới nền (Background Task)
    để tránh làm treo/ngắt kết nối API.
    """
    try:
        background_tasks.add_task(run_training_pipeline)
        return Response("Training process started in background!!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict", tags=["prediction"])
async def predict_route(request: TextRequest):
    """API thực hiện tóm tắt văn bản với tùy chọn độ dài"""
    try:
        if prediction_pipeline is None:
            return Response("Model chưa được nạp thành công!", status_code=500)

        # Gọi hàm predict và truyền cả nội dung lẫn lựa chọn độ dài (.value)
        summary = prediction_pipeline.predict(request.text, request.summary_type.value)
        return {"summary": summary, "summary_type": request.summary_type.value}

    except Exception as e:
        return Response(f"Lỗi khi dự đoán: {str(e)}", status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)