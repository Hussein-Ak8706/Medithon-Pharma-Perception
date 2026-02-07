from bs4 import BeautifulSoup
import requests

url = 'https://www.drugs.com/comments/codeine/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) Gecko/20100101 Firefox/147.0'}
result = requests.get(url, headers=headers)
decodedResult = result.content.decode()


soup = BeautifulSoup(decodedResult, 'html.parser')


reviews_2d = []

# find all review blocks
review_blocks = soup.find_all(
    class_="ddc-comment ddc-box ddc-mgb-2"
)

# find all rating summary blocks
rating_blocks = soup.find_all(
    class_="ddc-rating-summary ddc-mgb-1"
)

# iterate over both together (assumes same order & count)
for review, rating in zip(review_blocks, rating_blocks):
    
    review_text = review.get_text(strip=True)
    rating_text = rating.get_text(strip=True)
    
    reviews_2d.append([review_text, rating_text])

print(reviews_2d[0])
for i in reviews_2d:
    i[0] = i[0][6:-37]

print(reviews_2d)