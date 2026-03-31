"""
etl_pipeline.py
===============
Day 8: UK Property Price ETL + Trend Analysis

Industry:  Real Estate
Format:    Python script (.py)
Skills:    ETL, pandas, sqlite3, matplotlib, data cleaning

Who uses this:
    A property investor comparing area price trends before making
    purchase decisions. This pipeline transforms raw Land Registry
    transaction data into clean area-level price trend analysis.
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────
RAW_FILE   = 'land_registry_2024.csv'
DB_FILE    = 'property_prices.db'
OUTPUT_DIR = 'output'
CHART_FILE = 'property_price_analysis.png'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUMNS = [
    'transaction_id', 'price', 'date', 'postcode', 'property_type',
    'new_build', 'tenure', 'paon', 'saon', 'street',
    'locality', 'town', 'district', 'county', 'ppd_type', 'record_status'
]

PROPERTY_TYPE_MAP = {
    'D': 'Detached', 'S': 'Semi-Detached',
    'T': 'Terraced', 'F': 'Flat', 'O': 'Other'
}

print('=' * 60)
print('UK PROPERTY PRICE ETL PIPELINE')
print('=' * 60)


# ══════════════════════════════════════════════════════════════
# EXTRACT
# ══════════════════════════════════════════════════════════════
print('\n[EXTRACT] Loading Land Registry data...')
start = time.time()

df_raw = pd.read_csv(RAW_FILE, header=None, names=COLUMNS, low_memory=False)

print(f'  Raw rows loaded:  {len(df_raw):,}')
print(f'  Time taken:       {round(time.time()-start, 2)}s')


# ══════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════
print('\n[TRANSFORM] Cleaning and enriching data...')
start = time.time()

df = df_raw.copy()

# T1 — Convert price to numeric
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# T2 — Parse date
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# T3 — Drop nulls in key columns
rows_before = len(df)
df = df.dropna(subset=['price', 'date', 'town'])
print(f'  Dropped {rows_before - len(df):,} rows with missing key values')

# T4 — Remove outliers — prices below £10k or above £10M are errors
df = df[(df['price'] >= 10_000) & (df['price'] <= 10_000_000)]

# T5 — Standardise text columns
df['town']     = df['town'].str.strip().str.upper()
df['county']   = df['county'].str.strip().str.upper()
df['postcode'] = df['postcode'].str.strip().str.upper()

# T6 — Map property type codes to readable labels
df['property_type_label'] = df['property_type'].map(PROPERTY_TYPE_MAP).fillna('Other')

# T7 — Engineer useful features
df['year']        = df['date'].dt.year
df['month']       = df['date'].dt.month
df['quarter']     = df['date'].dt.quarter
df['year_quarter'] = df['year'].astype(str) + ' Q' + df['quarter'].astype(str)
df['is_new_build'] = (df['new_build'] == 'Y').astype(int)

# T8 — Extract postcode area (first 2-4 chars before space)
df['postcode_area'] = df['postcode'].str.split(' ').str[0].str[:4]

print(f'  Clean rows:       {len(df):,}')
print(f'  Date range:       {df["date"].min().date()} to {df["date"].max().date()}')
print(f'  Towns covered:    {df["town"].nunique():,}')
print(f'  Time taken:       {round(time.time()-start, 2)}s')


# ══════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════
print('\n[LOAD] Writing to SQLite...')
start = time.time()

conn = sqlite3.connect(DB_FILE)
df.to_sql('transactions', conn, if_exists='replace', index=False)

# Monthly summary table
monthly = (
    df.groupby(['town', 'year_quarter'])
    .agg(
        median_price=('price', 'median'),
        avg_price=('price', 'mean'),
        transaction_count=('price', 'count')
    )
    .round(0)
    .reset_index()
)
monthly.to_sql('monthly_summary', conn, if_exists='replace', index=False)

print(f'  transactions table:    {len(df):,} rows')
print(f'  monthly_summary table: {len(monthly):,} rows')
print(f'  Time taken:            {round(time.time()-start, 2)}s')


# ══════════════════════════════════════════════════════════════
# VALIDATE
# ══════════════════════════════════════════════════════════════
print('\n[VALIDATE] Running integrity checks...')

checks = {}

row_count = pd.read_sql_query('SELECT COUNT(*) as cnt FROM transactions', conn).iloc[0]['cnt']
checks['Row count matches'] = int(row_count) == len(df)

null_check = pd.read_sql_query(
    'SELECT COUNT(*) as cnt FROM transactions WHERE price IS NULL OR date IS NULL', conn
).iloc[0]['cnt']
checks['No nulls in key columns'] = int(null_check) == 0

price_check = pd.read_sql_query(
    'SELECT COUNT(*) as cnt FROM transactions WHERE price < 10000 OR price > 10000000', conn
).iloc[0]['cnt']
checks['All prices in valid range'] = int(price_check) == 0

town_check = pd.read_sql_query(
    'SELECT COUNT(DISTINCT town) as cnt FROM transactions', conn
).iloc[0]['cnt']
checks['Multiple towns present'] = int(town_check) > 10

for check, passed in checks.items():
    print(f'  [{"PASS" if passed else "FAIL"}] {check}')

all_passed = all(checks.values())
print(f'\n  Overall: {"ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"}')


# ══════════════════════════════════════════════════════════════
# ANALYSE
# ══════════════════════════════════════════════════════════════
print('\n[ANALYSE] Running SQL analysis queries...')

# Load full data into pandas for median calculations
df_sql = pd.read_sql_query('SELECT town, price, property_type_label, is_new_build, year_quarter FROM transactions', conn)

# Q1 — Top 15 most expensive towns by median price
town_stats = (
    df_sql.groupby('town')['price']
    .agg(['median', 'mean', 'count'])
    .rename(columns={'median': 'median_price', 'mean': 'avg_price', 'count': 'transactions'})
    .query('transactions >= 50')
    .round(0)
    .reset_index()
)
town_stats.columns = ['town', 'median_price', 'avg_price', 'transactions']

q1 = town_stats.sort_values('median_price', ascending=False).head(15)
q2 = town_stats.sort_values('median_price', ascending=True).head(15)

# Q3 — Price by property type
q3 = (
    df_sql.groupby('property_type_label')['price']
    .agg(['median', 'mean', 'count'])
    .rename(columns={'median': 'median_price', 'mean': 'avg_price', 'count': 'transactions'})
    .round(0)
    .sort_values('median_price', ascending=False)
    .reset_index()
)
q3.columns = ['property_type_label', 'median_price', 'avg_price', 'transactions']

# Q4 — New build vs existing
q4 = (
    df_sql.groupby(df_sql['is_new_build'].map({1: 'New Build', 0: 'Existing'}))['price']
    .agg(['median', 'mean', 'count'])
    .rename(columns={'median': 'median_price', 'mean': 'avg_price', 'count': 'transactions'})
    .round(0)
    .reset_index()
)
q4.columns = ['build_type', 'median_price', 'avg_price', 'transactions']

# Q5 — Transaction volume by quarter
q5 = (
    df_sql.groupby('year_quarter')
    .agg(transactions=('price', 'count'), avg_price=('price', 'mean'))
    .round(0)
    .reset_index()
    .sort_values('year_quarter')
)

print('  Queries complete')
print('\n=== Top 10 most expensive towns ===')
print(q1.head(10).to_string())
print('\n=== Price by property type ===')
print(q3.to_string())
print('\n=== New build vs existing ===')
print(q4.to_string())


# ══════════════════════════════════════════════════════════════
# VISUALISE
# ══════════════════════════════════════════════════════════════
print('\n[VISUALISE] Building dashboard...')

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('UK Property Price Analysis — Land Registry 2024',
             fontsize=14, fontweight='bold', y=1.01)

# Panel 1 — Top 15 most expensive towns
axes[0, 0].barh(q1['town'], q1['median_price'] / 1000, color='#E24B4A')
axes[0, 0].set_xlabel('Median price (£ thousands)')
axes[0, 0].set_title('Top 15 most expensive towns')
axes[0, 0].invert_yaxis()
axes[0, 0].tick_params(axis='y', labelsize=8)

# Panel 2 — Top 15 most affordable towns
axes[0, 1].barh(q2['town'], q2['median_price'] / 1000, color='#1D9E75')
axes[0, 1].set_xlabel('Median price (£ thousands)')
axes[0, 1].set_title('Top 15 most affordable towns')
axes[0, 1].invert_yaxis()
axes[0, 1].tick_params(axis='y', labelsize=8)

# Panel 3 — Price by property type
pt_colors = ['#378ADD', '#1D9E75', '#EF9F27', '#E24B4A', '#888780']
bars = axes[1, 0].bar(
    q3['property_type_label'],
    q3['median_price'] / 1000,
    color=pt_colors[:len(q3)]
)
axes[1, 0].set_ylabel('Median price (£ thousands)')
axes[1, 0].set_title('Median price by property type')
axes[1, 0].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, q3['median_price']):
    axes[1, 0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f'£{val/1000:.0f}k',
        ha='center', va='bottom', fontsize=9, fontweight='bold'
    )

# Panel 4 — Transaction volume by quarter
if len(q5) > 1:
    ax4b = axes[1, 1].twinx()
    axes[1, 1].bar(q5['year_quarter'], q5['transactions'], color='#B5D4F4', alpha=0.7)
    ax4b.plot(q5['year_quarter'], q5['avg_price'] / 1000,
              color='#E24B4A', linewidth=2, marker='o', markersize=5)
    axes[1, 1].set_ylabel('Transaction count', color='#378ADD')
    ax4b.set_ylabel('Avg price (£ thousands)', color='#E24B4A')
    axes[1, 1].set_title('Transaction volume + avg price by quarter')
    axes[1, 1].tick_params(axis='x', rotation=20)
else:
    axes[1, 1].bar(['2024'], [q5['transactions'].sum()], color='#B5D4F4')
    axes[1, 1].set_title('Total transactions 2024')

plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150, bbox_inches='tight')
print(f'  Chart saved as {CHART_FILE}')
plt.show()


# ══════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════
q1.to_csv(f'{OUTPUT_DIR}/most_expensive_towns.csv', index=False)
q2.to_csv(f'{OUTPUT_DIR}/most_affordable_towns.csv', index=False)
q3.to_csv(f'{OUTPUT_DIR}/price_by_property_type.csv', index=False)
q4.to_csv(f'{OUTPUT_DIR}/new_build_vs_existing.csv', index=False)

conn.close()

total_value  = df['price'].sum()
median_price = df['price'].median()
most_exp     = q1.iloc[0]
most_aff     = q2.iloc[0]
new_build    = q4[q4['build_type'] == 'New Build'].iloc[0]
existing     = q4[q4['build_type'] == 'Existing'].iloc[0]
premium_pct  = round((new_build['median_price'] - existing['median_price']) / existing['median_price'] * 100, 1)

print('\n' + '=' * 60)
print('BUSINESS INSIGHT SUMMARY')
print('=' * 60)
print(f'Total transactions:      {len(df):,}')
print(f'Total market value:      £{total_value:,.0f}')
print(f'National median price:   £{median_price:,.0f}')
print(f'Towns analysed:          {df["town"].nunique():,}')
print()
print(f'Most expensive town:     {most_exp["town"]} — £{most_exp["median_price"]:,.0f} median')
print(f'Most affordable town:    {most_aff["town"]} — £{most_aff["median_price"]:,.0f} median')
print()
print(f'New build premium:       {premium_pct}% above existing properties')
print(f'New build median:        £{new_build["median_price"]:,.0f}')
print(f'Existing median:         £{existing["median_price"]:,.0f}')
print()
print('Most expensive property type:')
print(f'  {q3.iloc[0]["property_type_label"]} — £{q3.iloc[0]["median_price"]:,.0f} median')
print('Most affordable property type:')
print(f'  {q3.iloc[-1]["property_type_label"]} — £{q3.iloc[-1]["median_price"]:,.0f} median')
print()
print('Investor recommendation:')
print(f'  Target {most_aff["town"]} for entry-level yield plays')
print(f'  Avoid new builds — {premium_pct}% premium with no rental yield advantage')
print('=' * 60)
