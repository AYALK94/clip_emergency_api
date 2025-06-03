from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from clip_model import predict_combined

app = FastAPI(
    title="Unified Emergency Classifier",
    description="Upload an image (and optional text) to classify emergency severity using CLIP.",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
def home():
    return {"message": "✅ API is running"}

@app.post("/predict", tags=["Unified Classification"])
async def predict(file: UploadFile = File(...), description: str = Form(None)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG/PNG files are allowed.")

    try:
        result = predict_combined(file.file, description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))











# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from clip_model import classify_image, match_image_to_text
# import os
# import uuid
# from datetime import datetime

# app = FastAPI(
#     title="CLIP Emergency API",
#     description="Upload an image and description to classify or match using OpenAI CLIP.",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.get("/", tags=["Health"])
# def home():
#     return {"message": "✅ CLIP Emergency Classifier is running"}

# @app.post("/classify", tags=["Classification"])
# async def classify(file: UploadFile = File(...)):
#     if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         raise HTTPException(status_code=400, detail="Only JPG/PNG files allowed.")
    
#     try:
#         result = classify_image(file.file)
#         return JSONResponse(content=result)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/match", tags=["Image + Text Match"])
# async def match(
#     file: UploadFile = File(...),
#     description: str = Form(...)
# ):
#     if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         raise HTTPException(status_code=400, detail="Only JPG/PNG files allowed.")
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_ext = os.path.splitext(file.filename)[-1]
#     unique_name = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_ext}"
#     save_path = os.path.join(UPLOAD_DIR, unique_name)

#     with open(save_path, "wb") as buffer:
#         buffer.write(await file.read())

#     with open(save_path, "rb") as image_for_clip:
#         try:
#             result = match_image_to_text(image_for_clip, description)
#             result["saved_filename"] = unique_name
#             result["description_input"] = description
#             return JSONResponse(content=result)
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))