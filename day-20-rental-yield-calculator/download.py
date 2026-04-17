import requests, pandas as pd

url = 'https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/bulletins/indexofprivatehousingrentalprices/february2024/a2rentalgrowthbyregion'
r = requests.get(url)
print('Status:', r.status_code, '| Size:', len(r.content), 'bytes')
print(r.text[:500])