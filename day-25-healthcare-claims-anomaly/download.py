import requests, pandas as pd, time

BASE_URL = 'https://data.cms.gov/data-api/v1/dataset/8889d81e-2ee7-448f-8713-f071038289b5/data'
TOTAL    = 150000
LIMIT    = 1000

all_rows = []
for offset in range(0, TOTAL, LIMIT):
    r = requests.get(BASE_URL, params={'limit': LIMIT, 'offset': offset})
    batch = r.json()
    all_rows.extend(batch)
    print(f'  Fetched offset {offset} — {len(all_rows):,} rows so far')
    time.sleep(0.3)

df = pd.DataFrame(all_rows)
df.to_csv('claims.csv', index=False)
print(f'\nDone. Rows: {len(df):,} | Columns: {len(df.columns)}')