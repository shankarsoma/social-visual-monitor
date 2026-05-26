from playwright.sync_api import sync_playwright

def fetch_linkedin_posts(keyword):

    results = []

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page()

        page.goto("https://www.linkedin.com")

        results.append({
            "platform":"LinkedIn",
            "keyword":keyword,
            "status":"Scraper Connected"
        })

        browser.close()

    return results