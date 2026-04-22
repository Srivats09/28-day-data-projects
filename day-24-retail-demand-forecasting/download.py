import requests, pandas as pd

urls = [
    'https://raw.githubusercontent.com/sushantag9/Supermarket-Sales-Data-Analysis/master/supermarket_sales%20-%20Sheet1.csv',
    'https://raw.githubusercontent.com/RISHIshrivas/Retail-Sales-data-analysis/main/retail_sales_dataset.csv',
]

for url in urls:
    r = requests.get(url)
    print(f'Status: {r.status_code} | Size: {len(r.content)} | URL: {url[:60]}')
    if r.status_code == 200:
        open('sales.csv', 'wb').write(r.content)
        df = pd.read_csv('sales.csv')
        print(f'Rows: {len(df)} | Cols: {df.columns.tolist()}')
        print(df.head(2).to_string())
        break