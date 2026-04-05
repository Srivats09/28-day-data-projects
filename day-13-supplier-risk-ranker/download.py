import requests

url = 'https://raw.githubusercontent.com/jrcinco/supply-chain-shipment-price-data/master/SCMS_Delivery_History_Dataset.csv'
r = requests.get(url)
open('scms_supply_chain.csv', 'wb').write(r.content)
print('Downloaded:', len(r.content), 'bytes')