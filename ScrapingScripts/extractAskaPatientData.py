from playwright.sync_api import sync_playwright
import csv
import os
import time
from urllib.parse import urlparse, parse_qs, unquote_plus

INPUT_CSV = "urls.csv"
OUTPUT_CSV = "extracted_data.csv"

def scrape_one_link_and_remove():
    start_time = time.perf_counter()

    # Read all remaining links
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("No links left to process.")
        return

    # Take the first URL
    url = rows[0][0]
    remaining_rows = rows[1:]

    # ---- EXTRACT DRUG NAME FROM URL ----
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    drug_name_raw = query_params.get("name", [""])[0]
    drug_name = unquote_plus(drug_name_raw)

    # Ensure output CSV exists
    if not os.path.isfile(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["drug_name", "rating", "side_effects", "comments"])

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        try:
            print(f"Scraping: {drug_name}")
            page.goto(url, timeout=60000)
            page.wait_for_load_state("domcontentloaded")

            # -------- EXTRACT DATA FROM TABLE --------
            rows = page.locator("table.ratingsTable tbody tr")

            ratings = []
            side_effects = []
            comments = []

            for i in range(1, rows.count()):  # skip header row
                row = rows.nth(i)
                cells = row.locator("td")

                rating = cells.nth(0).inner_text().strip()
                side_effect = cells.nth(2).inner_text().strip()
                comment = cells.nth(3).inner_text().strip()

                ratings.append(rating)
                side_effects.append(side_effect)
                comments.append(comment)

            rating_text = " | ".join(ratings)
            side_effects_text = " | ".join(side_effects)
            comments_text = " || ".join(comments)

            # -------- SAVE RESULT --------
            with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    drug_name,
                    rating_text,
                    side_effects_text,
                    comments_text
                ])

        except Exception as e:
            print(f"Error scraping {url}: {e}")

        finally:
            page.close()
            browser.close()

    # -------- REMOVE USED LINK FROM INPUT CSV --------
    with open(INPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(remaining_rows)

    elapsed = time.perf_counter() - start_time
    print("Done. URL removed from input CSV.")
    print(f"Finished in {elapsed:.2f} seconds")

if __name__ == "__main__":
    scrape_one_link_and_remove()
