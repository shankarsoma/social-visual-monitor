from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from detection import detect_logo_similarity
from ocr import extract_text_from_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Social Visual Monitor Running"}

@app.post("/extract-ocr")
async def extract_ocr(file: UploadFile = File(...)):

    contents = await file.read()

    result = extract_text_from_image(contents)

    return {
        "ocr_text": result
    }

@app.post("/detect-logo")
async def detect_logo(file: UploadFile = File(...)):

    contents = await file.read()

    similarity = detect_logo_similarity(contents)

    return {
        "similarity_score": similarity
    }
