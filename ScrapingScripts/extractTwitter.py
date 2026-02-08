from playwright.sync_api import sync_playwright
import pandas as pd
import time
import random
import csv

USERNAME = "User_123"
PASSWORD = "Pass_123"

DRUGS = {}

with open("AnalyticData.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        drug = row["drug_name"].strip().lower()

        DRUGS[drug] = [
            drug,                     # generic / normalized name
            f'"{drug}" lang:en'       # search query
        ]

print(DRUGS)

def human_sleep(a=1.5, b=3.5):
    time.sleep(random.uniform(a, b))

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=False,
        args=["--disable-blink-features=AutomationControlled"]
    )

    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    )

    page = context.new_page()

    # -------------------------
    # LOGIN
    # -------------------------
    page.goto("https://x.com/login")
    page.wait_for_timeout(5000)

    page.fill('input[name="text"]', USERNAME)
    page.keyboard.press("Enter")
    page.wait_for_timeout(3000)

    page.fill('input[name="password"]', PASSWORD)
    page.keyboard.press("Enter")
    page.wait_for_timeout(8000)

    # -------------------------
    # SCRAPING
    # -------------------------
    rows = []

    for brand, drug_search in DRUGS.items():
        search_url = f"https://x.com/search?q={drug_search[1]}&f=live"
        page.goto(search_url)
        page.wait_for_timeout(5000)

        # Scroll to load tweets
        for _ in range(6):
            page.mouse.wheel(0, 4000)
            human_sleep()

        tweets = page.query_selector_all("article")

        for tweet in tweets:
            try:
                text_block = tweet.query_selector('div[data-testid="tweetText"]')
                if text_block:
                    tweet_text = text_block.inner_text()

                    rows.append({
                        "brand_name": brand,
                        "generic_name": drug_search[0],
                        "tweet_text": tweet_text
                    })
            except:
                continue

        print(f"Collected tweets for {brand}")

    # -------------------------
    # DATAFRAME
    # -------------------------
    df = pd.DataFrame(rows)
    print(df.head())

    df.to_csv("drug_perception_tweets.csv", index=False)

    # Save session for reuse (IMPORTANT)
    context.storage_state(path="auth.json")

    browser.close()
