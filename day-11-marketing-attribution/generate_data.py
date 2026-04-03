import pandas as pd
import numpy as np
import random
random.seed(42)
np.random.seed(42)

channels = ['Google Search', 'Facebook', 'Email', 'Instagram', 'YouTube', 'Direct', 'Organic SEO']
n_customers = 2000

rows = []
for cust_id in range(1, n_customers + 1):
    n_touches = random.randint(1, 6)
    journey = random.choices(channels, k=n_touches)
    converted = 1 if random.random() < 0.35 else 0
    revenue = round(np.random.lognormal(4.5, 0.7), 2) if converted else 0
    for i, channel in enumerate(journey):
        rows.append({
            'customer_id': cust_id,
            'touchpoint_order': i + 1,
            'total_touchpoints': n_touches,
            'channel': channel,
            'converted': converted,
            'revenue': revenue if i == len(journey)-1 else 0
        })

df = pd.DataFrame(rows)
df.to_csv('customer_journeys.csv', index=False)
print(f'Generated {len(df)} touchpoints for {n_customers} customers')
print(df['channel'].value_counts().to_string())