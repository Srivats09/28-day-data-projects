import requests, zipfile, io, pandas as pd

url = 'https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip'
r = requests.get(url, timeout=60)
outer_zip = zipfile.ZipFile(io.BytesIO(r.content))

# Get the inner zip file
inner_zip_name = [f for f in outer_zip.namelist() if f.endswith('.zip')][0]
inner_zip_data = outer_zip.read(inner_zip_name)
inner_zip = zipfile.ZipFile(io.BytesIO(inner_zip_data))

print('Files in inner zip:', inner_zip.namelist())
inner_zip.extractall('nasa_data')
print('Extracted successfully')

# Preview train_FD001.txt
cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
df = pd.read_csv('nasa_data/train_FD001.txt', sep='\s+', header=None, names=cols)
print(f'Rows: {len(df):,} | Columns: {len(df.columns)}')
print(df.head(3).to_string())