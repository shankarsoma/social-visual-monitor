from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import os
import json
import pandas as pd
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash

app = FastAPI()

# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# LOGO STATE — persisted to /tmp so it
# survives across workers on Render
# ==========================================

LOGO_STATE_FILE = "/tmp/logo_state.json"
UPLOAD_DIR      = "/tmp/uploads"
TEMP_IMAGE      = "/tmp/temp_image.jpg"

def get_logo_path():
    try:
        with open(LOGO_STATE_FILE, "r") as f:
            state = json.load(f)
            path = state.get("logo_path")
            if path and os.path.exists(path):
                return path
    except Exception:
        pass
    return None

def set_logo_path(path):
    with open(LOGO_STATE_FILE, "w") as f:
        json.dump({"logo_path": path}, f)

# ==========================================
# SCORING — SSIM + ImageHash combined
# ==========================================

def compute_ssim_score(logo_gray, image_gray):
    """Structural similarity — catches colour/shape match."""
    try:
        resized = cv2.resize(image_gray, (logo_gray.shape[1], logo_gray.shape[0]))
        score, _ = ssim(logo_gray, resized, full=True)
        return round(max(score, 0) * 100, 2)
    except Exception:
        return 0.0

def compute_hash_score(logo_pil, image_pil):
    """Perceptual hash — catches reposted/resaved creatives."""
    try:
        h1 = imagehash.phash(logo_pil)
        h2 = imagehash.phash(image_pil)
        diff = h1 - h2          # hamming distance 0–64
        score = max(0, (64 - diff) / 64 * 100)
        return round(score, 2)
    except Exception:
        return 0.0

def combined_score(ssim_score, hash_score):
    """
    Weighted blend — hash is weighted higher because it is more
    robust to resizing and JPEG re-compression used on social media.
    SSIM still matters for colour and structural layout.
    """
    return round(ssim_score * 0.35 + hash_score * 0.65, 2)

def risk_level(score):
    if score >= 65:
        return "High"
    elif score >= 35:
        return "Medium"
    else:
        return "Low"

def compare_logo(logo_path, image_path):
    """
    Returns (ssim_score, hash_score, combined, risk).
    All scores are 0–100 floats.
    """
    try:
        logo_cv  = cv2.imread(logo_path)
        image_cv = cv2.imread(image_path)
        if logo_cv is None or image_cv is None:
            return 0.0, 0.0, 0.0, "Low"

        logo_gray  = cv2.cvtColor(logo_cv,  cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        logo_pil  = Image.open(logo_path).convert("RGB")
        image_pil = Image.open(image_path).convert("RGB")

        s = compute_ssim_score(logo_gray, image_gray)
        h = compute_hash_score(logo_pil, image_pil)
        c = combined_score(s, h)
        r = risk_level(c)

        return s, h, c, r

    except Exception as e:
        print("COMPARE ERROR:", e)
        return 0.0, 0.0, 0.0, "Low"

def download_image(url):
    try:
        r = requests.get(url, timeout=15)
        with open(TEMP_IMAGE, "wb") as f:
            f.write(r.content)
        return TEMP_IMAGE
    except Exception:
        return None

def fmt_date(raw):
    """Convert ISO timestamp → human readable IST."""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %I:%M %p IST")
    except Exception:
        return raw or "—"

def now_ist():
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %I:%M %p IST")

# ==========================================
# BRAND + PLATFORM CONFIGURATION
#
# For each brand, add dataset_id per platform.
# Leave the value as "" if you have not yet
# set up that platform in Apify — the scan
# will skip it gracefully.
# ==========================================

BRANDS = {
    "ICICI": {
        "instagram": "kEaw2FDvOje5FYEH8",
        "facebook":  "",
        "linkedin":  "",
    },
    "Groww": {
        "instagram": "UXNdHqAR3NV84I4yK",
        "facebook":  "",
        "linkedin":  "",
    },
    "Motilal Oswal": {
        "instagram": "jxqI2FhmwKxANA7ID",
        "facebook":  "",
        "linkedin":  "",
    },
    "Tata Capital": {
        "instagram": "0lPf4mavgwrO3Htde",
        "facebook":  "",
        "linkedin":  "",
    },
    "Zerodha": {
        "instagram": "nHeWy0j1RLe9eyKVD",
        "facebook":  "",
        "linkedin":  "",
    },
    "Upstox": {
        "instagram": "s2xel18LTiGzunXjS",
        "facebook":  "",
        "linkedin":  "",
    },
    "SBI": {
        "instagram": "bwXcVBwdWc6OvnSmU",
        "facebook":  "",
        "linkedin":  "",
    },
    "Anand Rathi": {
        "instagram": "WaccQaOU5HTHAgRAr",
        "facebook":  "",
        "linkedin":  "",
    },
}

PLATFORMS = ["instagram", "facebook", "linkedin"]

PLATFORM_DISPLAY = {
    "instagram": "Instagram",
    "facebook":  "Facebook",
    "linkedin":  "LinkedIn",
}

# ==========================================
# RISK COLOUR HELPERS
# ==========================================

RISK_COLOR = {
    "High":   "#dc3545",
    "Medium": "#fd7e14",
    "Low":    "#198754",
}

def risk_badge(level):
    color = RISK_COLOR.get(level, "#333")
    return f'<span style="color:{color};font-weight:600">{level}</span>'

# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():
    return {"status": "AI Visual Threat Monitoring Running"}

# ==========================================
# UPLOAD LOGO
# ==========================================

@app.post("/upload")
async def upload_logo(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    set_logo_path(filepath)
    return {
        "message": "Logo uploaded successfully",
        "file":    filepath,
        "status":  "ready_to_scan",
    }

# ==========================================
# LOGO STATUS
# ==========================================

@app.get("/logo-status")
def logo_status():
    path = get_logo_path()
    if path:
        return {"uploaded": True, "filename": os.path.basename(path)}
    return {"uploaded": False}

# ==========================================
# DASHBOARD
# ==========================================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    logo_path     = get_logo_path()
    logo_filename = os.path.basename(logo_path) if logo_path else None

    logo_status_html = (
        f'<div class="banner banner-ok">✅ Logo loaded: <strong>{logo_filename}</strong> — ready to scan</div>'
        if logo_path else
        '<div class="banner banner-warn">⚠️ No logo uploaded yet. Upload one below before scanning.</div>'
    )

    brand_options = "".join(
        f'<option value="{b}">{b}</option>' for b in BRANDS
    )

    platform_options = "".join(
        f'<option value="{p}">{PLATFORM_DISPLAY[p]}</option>' for p in PLATFORMS
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <title>AI Visual Threat Monitoring</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:Arial,sans-serif;background:#f5f6fa;padding:32px;color:#222}}
    h1{{font-size:22px;margin-bottom:6px}}
    .sub{{color:#666;font-size:14px;margin-bottom:24px}}
    .banner{{padding:12px 16px;border-radius:6px;font-size:14px;margin-bottom:20px}}
    .banner-ok{{background:#d1fae5;border:1px solid #6ee7b7;color:#065f46}}
    .banner-warn{{background:#fef3c7;border:1px solid #fcd34d;color:#92400e}}
    .card{{background:#fff;border-radius:8px;padding:24px;margin-bottom:20px;
           box-shadow:0 1px 4px rgba(0,0,0,.08)}}
    .card h2{{font-size:16px;margin-bottom:8px;color:#333}}
    .card p{{font-size:14px;color:#666;margin-bottom:14px}}
    input[type=file]{{display:block;margin:10px 0;font-size:14px}}
    select{{padding:10px;font-size:14px;border:1px solid #ccc;border-radius:4px;min-width:220px}}
    button{{padding:10px 20px;font-size:14px;border:none;border-radius:4px;cursor:pointer;color:#fff}}
    .btn-green{{background:#198754}} .btn-green:hover{{background:#157347}}
    .btn-blue{{background:#0d6efd;margin-left:10px}} .btn-blue:hover{{background:#0b5ed7}}
    .row{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:4px}}
    #upload-result{{margin-top:10px;font-size:13px;color:#333}}
    .step-num{{display:inline-block;background:#0d6efd;color:#fff;border-radius:50%;
               width:22px;height:22px;text-align:center;line-height:22px;
               font-size:12px;margin-right:8px}}
  </style>
</head>
<body>
  <h1>🛡️ AI Visual Threat Monitoring</h1>
  <p class="sub">Brand logo comparison across Instagram, Facebook and LinkedIn</p>

  {logo_status_html}

  <!-- Step 1 -->
  <div class="card">
    <h2><span class="step-num">1</span> Upload Brand Logo</h2>
    <p>Upload the official logo image you want to compare against social media posts.</p>
    <input type="file" id="logoFile" accept="image/*">
    <button class="btn-green" onclick="uploadLogo()">Upload Logo</button>
    <div id="upload-result"></div>
  </div>

  <!-- Step 2 -->
  <div class="card">
    <h2><span class="step-num">2</span> Scan a Brand</h2>
    <p>Pick a brand and platform, then click Scan. The system will compare every post image against your uploaded logo and show match scores.</p>
    <form action="/scan" method="get">
      <div class="row">
        <select name="brand">{brand_options}</select>
        <select name="platform">{platform_options}</select>
        <button type="submit" class="btn-blue">Scan Brand</button>
      </div>
    </form>
  </div>

  <script>
    async function uploadLogo() {{
      const fileInput = document.getElementById('logoFile');
      const resultDiv = document.getElementById('upload-result');
      if (!fileInput.files.length) {{
        resultDiv.innerHTML = '⚠️ Please select a file first.';
        return;
      }}
      resultDiv.innerHTML = 'Uploading…';
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      try {{
        const res  = await fetch('/upload', {{method:'POST', body: formData}});
        const data = await res.json();
        if (data.status === 'ready_to_scan') {{
          resultDiv.innerHTML = '✅ Logo uploaded: <strong>' + fileInput.files[0].name + '</strong>. You can now scan.';
          setTimeout(() => location.reload(), 1200);
        }} else {{
          resultDiv.innerHTML = '❌ Upload failed: ' + JSON.stringify(data);
        }}
      }} catch(err) {{
        resultDiv.innerHTML = '❌ Error: ' + err.message;
      }}
    }}
  </script>
</body>
</html>"""

# ==========================================
# SHARED: fetch + score posts from one
# Apify dataset
# ==========================================

def fetch_and_score(dataset_id, platform, brand, logo_path, apify_token):
    """
    Fetches posts from Apify, runs SSIM + ImageHash comparison,
    returns list of detection dicts.
    """
    url = (
        f"https://api.apify.com/v2/datasets/"
        f"{dataset_id}/items?token={apify_token}"
    )
    response = requests.get(url, timeout=30)
    data     = response.json()

    if not isinstance(data, list):
        return [], 0

    detections = []

    for post in data:
        try:
            caption       = str(post.get("caption", "")).replace("\n", " ")
            raw_date      = post.get("timestamp", "")
            published_date = fmt_date(raw_date)

            # Platform-specific image URL field
            if platform == "instagram":
                image_url = post.get("displayUrl", "") or post.get("imageUrl", "")
            elif platform == "facebook":
                image_url = post.get("media", [{}])[0].get("url", "") if post.get("media") else post.get("imageUrl", "")
            else:  # linkedin
                image_url = post.get("image", "") or post.get("imageUrl", "")

            # Platform-specific username / post URL
            if platform == "instagram":
                username = post.get("ownerUsername", "unknown")
                post_url = post.get("url", "#")
            elif platform == "facebook":
                username = post.get("pageName", "") or post.get("authorName", "unknown")
                post_url = post.get("url", "#") or post.get("postUrl", "#")
            else:  # linkedin
                username = post.get("authorName", "") or post.get("companyName", "unknown")
                post_url = post.get("url", "#") or post.get("postUrl", "#")

            ssim_s = hash_s = comb_s = 0.0
            risk   = "Low"

            if logo_path and image_url:
                img_path = download_image(image_url)
                if img_path:
                    ssim_s, hash_s, comb_s, risk = compare_logo(logo_path, img_path)

            detections.append({
                "platform":      PLATFORM_DISPLAY[platform],
                "username":      username,
                "publishedDate": published_date,
                "postUrl":       post_url,
                "detectedBrand": brand,
                "ssimScore":     ssim_s,
                "hashScore":     hash_s,
                "matchScore":    comb_s,
                "risk":          risk,
                "description": (
                    caption[:120] + "…"
                    if len(caption) > 120
                    else caption
                ),
            })

        except Exception as item_error:
            print(f"ITEM ERROR [{platform}]:", item_error)

    return detections, len(data)

# ==========================================
# SCAN
# ==========================================

@app.get("/scan", response_class=HTMLResponse)
def scan_instagram(brand: str, platform: str = "instagram"):

    APIFY_TOKEN = os.getenv("APIFY_TOKEN")
    if not APIFY_TOKEN:
        return "<h2>Missing APIFY_TOKEN environment variable</h2>"

    if brand not in BRANDS:
        return f"<h2>Unknown brand: {brand}</h2>"

    if platform not in PLATFORMS:
        return f"<h2>Unknown platform: {platform}</h2>"

    dataset_id = BRANDS[brand].get(platform, "")
    if not dataset_id:
        return f"""
        <h2 style='font-family:Arial;padding:40px'>
          No {PLATFORM_DISPLAY[platform]} dataset configured for {brand} yet.<br>
          <a href='/dashboard' style='font-size:16px'>← Back to Dashboard</a>
        </h2>"""

    logo_path = get_logo_path()

    try:
        detections, total_records = fetch_and_score(
            dataset_id, platform, brand, logo_path, APIFY_TOKEN
        )

        # Sort highest match first
        detections.sort(key=lambda x: x["matchScore"], reverse=True)

        # Risk counts
        high   = sum(1 for d in detections if d["risk"] == "High")
        medium = sum(1 for d in detections if d["risk"] == "Medium")
        low    = sum(1 for d in detections if d["risk"] == "Low")

        logo_banner = (
            f'<div class="banner banner-ok">✅ Logo used: <strong>{os.path.basename(logo_path)}</strong>'
            f' &nbsp;|&nbsp; Scoring: SSIM × 35% + ImageHash × 65%</div>'
            if logo_path else
            '<div class="banner banner-warn">⚠️ No logo uploaded — all scores are 0%. '
            '<a href="/dashboard">Upload a logo</a> first.</div>'
        )

        rows = ""
        for item in detections:
            rows += f"""
            <tr>
              <td>{item['platform']}</td>
              <td><strong>{item['username']}</strong></td>
              <td>{item['publishedDate']}</td>
              <td><span style="color:#0d6efd;font-weight:600">{item['detectedBrand']}</span></td>
              <td style="font-size:13px">{item['description']}</td>
              <td style="font-size:13px;color:#666">
                SSIM: {item['ssimScore']}%<br>
                Hash: {item['hashScore']}%
              </td>
              <td style="font-weight:700;font-size:15px">{item['matchScore']}%</td>
              <td>{risk_badge(item['risk'])}</td>
              <td><a href="{item['postUrl']}" target="_blank">View</a></td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
  <title>{brand} — {PLATFORM_DISPLAY[platform]}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:Arial,sans-serif;background:#f8f9fa;padding:28px;color:#222}}
    h1{{font-size:20px;margin-bottom:12px}}
    .banner{{padding:10px 14px;border-radius:6px;font-size:13px;margin-bottom:14px}}
    .banner-ok{{background:#d1fae5;border:1px solid #6ee7b7;color:#065f46}}
    .banner-warn{{background:#fef3c7;border:1px solid #fcd34d;color:#92400e}}
    .meta{{font-size:14px;color:#555;margin-bottom:16px;line-height:2}}
    .summary{{display:flex;gap:12px;margin-bottom:18px;flex-wrap:wrap}}
    .scard{{background:#fff;border-radius:6px;padding:12px 20px;min-width:130px;
            box-shadow:0 1px 3px rgba(0,0,0,.08);text-align:center}}
    .scard .num{{font-size:26px;font-weight:700}}
    .scard .lbl{{font-size:12px;color:#666;margin-top:2px}}
    .actions{{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap}}
    .btn{{padding:8px 16px;border-radius:4px;border:none;cursor:pointer;
          font-size:13px;text-decoration:none;display:inline-block;color:#fff}}
    .btn-gray{{background:#6c757d}} .btn-green{{background:#198754}}
    table{{width:100%;border-collapse:collapse;background:#fff;
           box-shadow:0 1px 4px rgba(0,0,0,.08);font-size:13px}}
    th{{background:#0d6efd;color:#fff;padding:11px 10px;text-align:left;font-size:13px}}
    td{{padding:9px 10px;border-bottom:1px solid #eee;vertical-align:top}}
    tr:hover td{{background:#f5f8ff}}
    a{{color:#0d6efd;text-decoration:none}}
  </style>
</head>
<body>
  <h1>🛡️ {brand} — {PLATFORM_DISPLAY[platform]} Monitoring</h1>

  {logo_banner}

  <div class="meta">
    Brand: <strong>{brand}</strong> &nbsp;|&nbsp;
    Platform: <strong>{PLATFORM_DISPLAY[platform]}</strong> &nbsp;|&nbsp;
    Dataset records: <strong>{total_records}</strong> &nbsp;|&nbsp;
    Last updated: <strong>{now_ist()}</strong>
  </div>

  <div class="summary">
    <div class="scard"><div class="num">{len(detections)}</div><div class="lbl">Total Posts</div></div>
    <div class="scard"><div class="num" style="color:#dc3545">{high}</div><div class="lbl">High Risk</div></div>
    <div class="scard"><div class="num" style="color:#fd7e14">{medium}</div><div class="lbl">Medium Risk</div></div>
    <div class="scard"><div class="num" style="color:#198754">{low}</div><div class="lbl">Low Risk</div></div>
  </div>

  <div class="actions">
    <a class="btn btn-gray" href="/dashboard">← Dashboard</a>
    <a class="btn btn-green" href="/export?brand={brand}&platform={platform}">⬇ Export Excel</a>
  </div>

  <table>
    <tr>
      <th>Platform</th><th>Username</th><th>Published</th><th>Brand</th>
      <th>Description</th><th>Score Breakdown</th><th>Match Score</th>
      <th>Risk</th><th>Post</th>
    </tr>
    {rows}
  </table>
</body>
</html>"""

        return html

    except Exception as e:
        print("SCAN ERROR:", e)
        return f"<h2>Error</h2><p>{e}</p>"

# ==========================================
# EXPORT EXCEL
# ==========================================

@app.get("/export")
def export_excel(brand: str, platform: str = "instagram"):

    APIFY_TOKEN = os.getenv("APIFY_TOKEN")
    if not APIFY_TOKEN:
        return {"error": "Missing APIFY_TOKEN"}

    if brand not in BRANDS:
        return {"error": f"Unknown brand: {brand}"}

    dataset_id = BRANDS[brand].get(platform, "")
    if not dataset_id:
        return {"error": f"No {platform} dataset configured for {brand}"}

    logo_path = get_logo_path()

    try:
        detections, _ = fetch_and_score(
            dataset_id, platform, brand, logo_path, APIFY_TOKEN
        )
        detections.sort(key=lambda x: x["matchScore"], reverse=True)

        rows = [{
            "Platform":         d["platform"],
            "Username":         d["username"],
            "Published Date":   d["publishedDate"],
            "Post URL":         d["postUrl"],
            "Detected Brand":   d["detectedBrand"],
            "SSIM Score (%)":   d["ssimScore"],
            "Hash Score (%)":   d["hashScore"],
            "Match Score (%)":  d["matchScore"],
            "Risk":             d["risk"],
            "Description":      d["description"],
        } for d in detections]

        df = pd.DataFrame(rows)

        # Summary sheet data
        high   = sum(1 for d in detections if d["risk"] == "High")
        medium = sum(1 for d in detections if d["risk"] == "Medium")
        low    = sum(1 for d in detections if d["risk"] == "Low")
        avg    = round(sum(d["matchScore"] for d in detections) / len(detections), 2) if detections else 0

        summary_df = pd.DataFrame([
            {"Metric": "Brand",             "Value": brand},
            {"Metric": "Platform",          "Value": PLATFORM_DISPLAY[platform]},
            {"Metric": "Scan Date (IST)",   "Value": now_ist()},
            {"Metric": "Total Posts",       "Value": len(detections)},
            {"Metric": "High Risk Posts",   "Value": high},
            {"Metric": "Medium Risk Posts", "Value": medium},
            {"Metric": "Low Risk Posts",    "Value": low},
            {"Metric": "Avg Match Score",   "Value": f"{avg}%"},
            {"Metric": "Logo File",         "Value": os.path.basename(logo_path) if logo_path else "None"},
            {"Metric": "Scoring Method",    "Value": "SSIM 35% + ImageHash 65%"},
        ])

        os.makedirs("/tmp/exports", exist_ok=True)
        timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S")
        filename  = f"/tmp/exports/{brand}_{platform}_{timestamp}.xlsx"

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:

            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            df.to_excel(writer, index=False, sheet_name="Detections")

            for sheet_name in ["Summary", "Detections"]:
                ws = writer.sheets[sheet_name]
                for col in ws.columns:
                    max_len = max(
                        (len(str(cell.value)) if cell.value else 0 for cell in col),
                        default=0
                    )
                    ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

        return FileResponse(
            path=filename,
            filename=f"{brand}_{platform}_{timestamp}.xlsx",
            media_type=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
        )

    except Exception as e:
        print("EXPORT ERROR:", e)
        return {"error": str(e)}
