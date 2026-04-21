import requests, zipfile, io, pandas as pd

url = 'https://analyse.kmi.open.ac.uk/open-dataset/download'

r = requests.get(url, timeout=30)
print('Status:', r.status_code, '| Size:', len(r.content), 'bytes')
if r.status_code == 200:
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print('Files in zip:', z.namelist())
    z.extractall('ou_data')
else:
    print(r.text[:200])