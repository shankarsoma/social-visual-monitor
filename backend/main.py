from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    return {
        "status": "AiPlex ORM API Running"
    }

@app.get("/scan")
def scan():

    return [
        {
            "platform":"Instagram",
            "username":"trading_master",
            "url":"https://www.instagram.com/p/DXXA7x3GfUw/",
            "brand":"Zerodha",
            "score":"94%",
            "risk":"Critical",
            "ocr":"Join VIP Telegram Group",
            "time":"2 hours ago"
        },
        {
            "platform":"Facebook",
            "username":"stocksignalsindia",
            "url":"https://www.facebook.com/countrysabsepahle/posts/pfbid02Nfacp5XtSrxiDTZ9PuUHCmyvPAMUjnL2S8dWPDqoeBwBmWaYFLT32QPFeLU2RdF7l",
            "brand":"Groww",
            "score":"89%",
            "risk":"High",
            "ocr":"Guaranteed Returns",
            "time":"1 day ago"
        },
        {
            "platform":"LinkedIn",
            "username":"Market Insights",
            "url":"https://www.linkedin.com/feed/update/urn:li:activity:7464967476284653568/",
            "brand":"ICICI Direct",
            "score":"91%",
            "risk":"Medium",
            "ocr":"Premium Signals Group",
            "time":"3 days ago"
        }
    ]