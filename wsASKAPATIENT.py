from playwright.sync_api import sync_playwright
import csv
import os

def collect_urls_no_click():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto("https://www.askapatient.com/drugalpha.asp?letter=A")

        # Extract hrefs directly
        urls = page.locator("table a").evaluate_all(
            "els => els.map(e => e.href)"
        )

        browser.close()

    # ---- CSV FIX STARTS HERE ----
    file_exists = os.path.isfile("urls_n.csv")

    with open("urls_n.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # write header only once
        if not file_exists:
            writer.writerow(["url"])

        for url in urls:
            writer.writerow([url])
    # ---- CSV FIX ENDS HERE ----

    return urls


if __name__ == "__main__":
    urls = collect_urls_no_click()
    print(f"Collected {len(urls)} URLs")
