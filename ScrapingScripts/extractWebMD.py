from playwright.sync_api import sync_playwright, TimeoutError
import csv
import time
import random
import string

OUTPUT_CSV = "webmd_reviews_all.csv"
BASE_URL = "https://www.webmd.com"

def get_drug_slugs(page, alpha_url):
    """Get all drug "slugs" from the alphabet page"""
    page.goto(alpha_url, wait_until="domcontentloaded")
    time.sleep(random.uniform(2, 4))
    page.locator("div.drugs-search-list-conditions").first.wait_for(timeout=15000)

    slugs = []
    for i in range(page.locator("a.alpha-drug-name").count()):
        href = page.locator("a.alpha-drug-name").nth(i).get_attribute("href")
        if href and href.startswith("/drugs/"):
            slug = href.split("/drugs/")[-1]  # get last part
            slugs.append(slug)
    return slugs

def scrape_reviews(page, review_url):
    """Scrape ratings and reviews from a WebMD review page"""
    page.goto(review_url, wait_until="domcontentloaded")
    time.sleep(random.uniform(2, 4))

    data_list = []

    # --- Get ratings ---
    ratings_section = page.locator(".ratings-section")
    rating_text = None
    if ratings_section.count() > 0:
        rating_text = ratings_section.inner_text().strip()

    # --- Get user reviews ---
    reviews_container = page.locator(".shared-reviews-container")
    try:
        reviews_container.first.wait_for(timeout=8000)
    except TimeoutError:
        return None

    review_items = reviews_container.locator("div")  # each review is inside a div
    for i in range(review_items.count()):
        review_text = review_items.nth(i).inner_text().strip()
        if review_text:
            data_list.append({
                "review_url": review_url,
                "rating": rating_text,
                "reviewText": review_text
            })

    return data_list if data_list else None

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["review_url", "rating", "reviewText"])
            writer.writeheader()

            # Loop through letters A-Z
            for letter in string.ascii_lowercase:
                alpha_url = f"{BASE_URL}/drugs/2/alpha/{letter}"
                print(f"\n--- Processing letter '{letter.upper()}' ---")
                try:
                    slugs = get_drug_slugs(page, alpha_url)
                    print(f"Found {len(slugs)} drugs for letter '{letter.upper()}'")
                except TimeoutError:
                    print(f"Failed to load drugs for letter '{letter.upper()}', skipping.")
                    continue

                for idx, slug in enumerate(slugs, start=1):
                    review_url = f"https://reviews.webmd.com/drugs/drugreview-{slug}"
                    print(f"[{idx}/{len(slugs)}] Scraping {review_url}")
                    try:
                        reviews = scrape_reviews(page, review_url)
                        if reviews:
                            for review in reviews:
                                writer.writerow(review)
                        else:
                            print("  -> No reviews found for this drug.")
                        time.sleep(random.uniform(1, 2))
                    except Exception as e:
                        print(f"  -> Error scraping {review_url}: {e}")
                        continue

        browser.close()
    print(f"\nAll reviews written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
