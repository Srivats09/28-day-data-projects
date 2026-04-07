import requests, zipfile, io

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('.')
print('Files extracted:')
import os
for f in os.listdir('.'):
    if f.endswith('.csv') or f.endswith('.txt'):
        print(f' ', f)