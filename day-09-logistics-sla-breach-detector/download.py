import requests

url = 'https://data.cityofchicago.org/resource/v6vf-nfxy.csv'
params = {'$limit': 50000}
r = requests.get(url, params=params)

with open('chicago_311.csv', 'wb') as f:
    f.write(r.content)

print('Downloaded:', len(r.content), 'bytes')
