import requests, pandas as pd

url = 'https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/8a4a771c-3929-4e56-93ad-cdf13219dea5/download/demanddata_2026.csv'
r = requests.get(url)
print('Status:', r.status_code, '| Size:', len(r.content), 'bytes')
if r.status_code == 200:
    open('demand_data.csv', 'wb').write(r.content)
    df = pd.read_csv('demand_data.csv')
    print(f'Rows: {len(df)}')
    print(f'Columns: {df.columns.tolist()}')
    print(df.head(3).to_string())
else:
    print(r.text[:200])