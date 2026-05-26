# Social Visual Brand Monitoring System

AI-powered ORM monitoring dashboard for:
- Instagram
- Facebook
- LinkedIn

## Features

- OCR text extraction
- Logo similarity detection
- Risk scoring
- Screenshot evidence
- CSV/XLSX export
- Enterprise dashboard UI

---

## Setup Guide

### 1. Clone Repository

git clone YOUR_GITHUB_REPO

---

### 2. Create Virtual Environment

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

Windows:
python -m venv venv
venv\Scripts\activate

---

### 3. Install Requirements

pip install -r requirements.txt

---

### 4. Install Tesseract OCR

Mac:
brew install tesseract

Windows:
https://github.com/tesseract-ocr/tesseract/releases

---

### 5. Start Backend

cd backend
uvicorn main:app --reload

---

### 6. Open Frontend

Open:
frontend/index.html

