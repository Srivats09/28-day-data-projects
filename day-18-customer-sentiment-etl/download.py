import requests, pandas as pd

url = 'https://raw.githubusercontent.com/JOTOR/Datasets/main/tripadvisor_hotel_reviews.csv'
r = requests.get(url)
print('Status:', r.status_code, '| Size:', len(r.content), 'bytes')
open('reviews.csv', 'wb').write(r.content)
df = pd.read_csv('reviews.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print(df.head(3).to_string())