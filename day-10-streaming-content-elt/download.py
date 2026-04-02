import requests

url = 'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/netflix_titles.csv'
r = requests.get(url)
open('netflix_titles.csv', 'wb').write(r.content)
print('Downloaded:', len(r.content), 'bytes')