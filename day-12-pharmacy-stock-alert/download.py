import requests, json, pandas as pd

url = 'https://api.fda.gov/drug/shortages.json?limit=1000'
r = requests.get(url)
data = r.json()

# Check structure of first result
print(json.dumps(data['results'][0], indent=2)[:1000])