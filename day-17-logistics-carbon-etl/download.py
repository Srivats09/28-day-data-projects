import requests, pandas as pd

url = 'https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1197564/conversion-factors-2023-condensed-set-flat-format.csv'
r = requests.get(url)
print('Status:', r.status_code, '| Size:', len(r.content), 'bytes')
open('beis_emissions_factors.csv', 'wb').write(r.content)
df = pd.read_csv('beis_emissions_factors.csv', encoding='latin1')
print(f'Rows: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print(df.head(3).to_string())