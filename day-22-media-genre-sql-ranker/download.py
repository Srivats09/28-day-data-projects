import requests, zipfile, io, pandas as pd

url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
r = requests.get(url)
print('Status:', r.status_code, '| Size:', len(r.content), 'bytes')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('.')
movies  = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
print(f'Movies:  {len(movies):,} rows — {movies.columns.tolist()}')
print(f'Ratings: {len(ratings):,} rows — {ratings.columns.tolist()}')
print(movies.head(3).to_string())
print(ratings.head(3).to_string())