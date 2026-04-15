import requests, pandas as pd

url = 'https://gender-pay-gap.service.gov.uk/viewing/download-data/2025'
r = requests.get(url)
print('Status:', r.status_code, '| Size:', len(r.content), 'bytes')
open('gender_pay_gap.csv', 'wb').write(r.content)
df = pd.read_csv('gender_pay_gap.csv')
print(f'Rows: {len(df)}')
print(df.columns.tolist())
print(df.head(2).to_string())