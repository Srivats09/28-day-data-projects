import requests

urls_to_try = [
    'https://raw.githubusercontent.com/mayankpujara/Financial-Sentiment-Analysis/main/data.csv',
    'https://raw.githubusercontent.com/nickmuchi/financial-news-sentiment-analysis/main/data/financial_phrasebank.csv',
    'https://raw.githubusercontent.com/suriyadeepan/financial-sentiment/master/data/train.csv',
]

for url in urls_to_try:
    r = requests.get(url)
    print(f'Status: {r.status_code} | Size: {len(r.content)} | URL: {url.split("/")[-1]}')
    if r.status_code == 200 and len(r.content) > 1000:
        open('financial_phrasebank.csv', 'wb').write(r.content)
        print('SUCCESS — saved as financial_phrasebank.csv')
        print(r.text[:300])
        break