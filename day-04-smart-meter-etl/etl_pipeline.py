"""
Day 4: Smart Meter Energy ETL Pipeline
=======================================
Industry:  Energy / Utilities
Format:    Python script (.py)
Skills:    ETL, pandas, sqlite3, matplotlib, data cleaning, data validation

Who uses this:
    An energy analyst running weekly consumption reports from raw meter data.
    This pipeline takes messy minute-level readings and produces a clean,
    queryable SQLite database with hourly aggregations ready for analysis.

ETL Pattern:
    EXTRACT  → Load raw semicolon-separated meter data in chunks (2M+ rows)
    TRANSFORM → Clean nulls, fix types, engineer features, resample to hourly
    LOAD      → Insert clean data into SQLite with schema validation
    VALIDATE  → Run integrity checks and print a validation report
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import os
import time

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_FILE    = 'household_power_consumption.txt'
DB_FILE     = 'smart_meter.db'
OUTPUT_DIR  = 'output'
CHART_FILE  = 'consumption_analysis.png'
CHUNK_SIZE  = 100_000   # process 100k rows at a time — memory efficient

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 60)
print('SMART METER ETL PIPELINE')
print('=' * 60)


# ══════════════════════════════════════════════════════════════
# EXTRACT
# Read raw data in chunks — 2M+ rows, too large to load at once
# ══════════════════════════════════════════════════════════════
print('\n[EXTRACT] Loading raw smart meter data...')
start = time.time()

chunks = []
chunk_count = 0
total_rows = 0

for chunk in pd.read_csv(
    RAW_FILE,
    sep=';',
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    chunks.append(chunk)
    chunk_count += 1
    total_rows += len(chunk)
    print(f'  Chunk {chunk_count}: {len(chunk):,} rows loaded')

df_raw = pd.concat(chunks, ignore_index=True)

extract_time = round(time.time() - start, 2)
print(f'\n[EXTRACT COMPLETE]')
print(f'  Total rows:    {len(df_raw):,}')
print(f'  Columns:       {df_raw.columns.tolist()}')
print(f'  Time taken:    {extract_time}s')
print(f'  Missing values before clean:')
print(df_raw.isnull().sum().to_string())


# ══════════════════════════════════════════════════════════════
# TRANSFORM
# Clean, fix types, engineer features, resample to hourly
# ══════════════════════════════════════════════════════════════
print('\n[TRANSFORM] Cleaning and transforming data...')
start = time.time()

df = df_raw.copy()

# Step T1 — Replace '?' with NaN (this dataset uses ? for missing values)
df.replace('?', np.nan, inplace=True)

# Step T2 — Convert numeric columns
numeric_cols = [
    'Global_active_power', 'Global_reactive_power',
    'Voltage', 'Global_intensity',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step T3 — Parse datetime (combine Date + Time columns)
df['datetime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'],
    format='%d/%m/%Y %H:%M:%S',
    errors='coerce'
)

# Step T4 — Drop rows with null datetime or null active power
rows_before = len(df)
df = df.dropna(subset=['datetime', 'Global_active_power'])
rows_dropped = rows_before - len(df)
print(f'  Dropped {rows_dropped:,} rows with missing datetime or power reading')

# Step T5 — Fill remaining nulls with forward fill (sensor gap filling)
df[numeric_cols] = df[numeric_cols].ffill()

# Step T6 — Remove physical impossibilities
df = df[df['Global_active_power'] >= 0]
df = df[df['Voltage'].between(200, 260)]  # normal household voltage range

# Step T7 — Engineer useful features
df['year']         = df['datetime'].dt.year
df['month']        = df['datetime'].dt.month
df['day']          = df['datetime'].dt.day
df['hour']         = df['datetime'].dt.hour
df['day_of_week']  = df['datetime'].dt.dayofweek  # 0=Mon, 6=Sun
df['day_name']     = df['datetime'].dt.day_name()
df['is_weekend']   = df['day_of_week'].isin([5, 6]).astype(int)

# Step T8 — Calculate total sub-metering and unaccounted power
df['total_sub_metering'] = (
    df['Sub_metering_1'] +
    df['Sub_metering_2'] +
    df['Sub_metering_3']
)
# Active energy consumed per minute (Wh) = power (kW) * (1/60 hr) * 1000
df['active_energy_wh'] = df['Global_active_power'] * (1000 / 60)

# Step T9 — Resample to hourly aggregations
print('  Resampling minute data to hourly aggregations...')
df = df.set_index('datetime')

df_hourly = df.resample('h').agg(
    avg_active_power=('Global_active_power', 'mean'),
    max_active_power=('Global_active_power', 'max'),
    total_energy_wh=('active_energy_wh', 'sum'),
    avg_voltage=('Voltage', 'mean'),
    avg_intensity=('Global_intensity', 'mean'),
    sub_meter_1_wh=('Sub_metering_1', 'sum'),
    sub_meter_2_wh=('Sub_metering_2', 'sum'),
    sub_meter_3_wh=('Sub_metering_3', 'sum'),
    is_weekend=('is_weekend', 'first'),
    day_name=('day_name', 'first'),
    hour=('hour', 'first'),
    year=('year', 'first'),
    month=('month', 'first')
).reset_index()

df_hourly = df_hourly.round(4)
df_hourly = df_hourly.dropna(subset=['avg_active_power'])

transform_time = round(time.time() - start, 2)
print(f'\n[TRANSFORM COMPLETE]')
print(f'  Clean minute rows:  {len(df):,}')
print(f'  Hourly aggregations: {len(df_hourly):,}')
print(f'  Date range: {df_hourly["datetime"].min()} to {df_hourly["datetime"].max()}')
print(f'  Time taken: {transform_time}s')


# ══════════════════════════════════════════════════════════════
# LOAD
# Insert clean hourly data into SQLite
# ══════════════════════════════════════════════════════════════
print('\n[LOAD] Writing to SQLite database...')
start = time.time()

conn = sqlite3.connect(DB_FILE)

# Write hourly table
df_hourly.to_sql('hourly_consumption', conn, if_exists='replace', index=False)

# Write a monthly summary table
df_hourly['year_month'] = df_hourly['year'].astype(str) + '-' + df_hourly['month'].astype(str).str.zfill(2)
monthly = df_hourly.groupby('year_month').agg(
    total_kwh=('total_energy_wh', lambda x: round(x.sum() / 1000, 2)),
    avg_hourly_power=('avg_active_power', lambda x: round(x.mean(), 3)),
    peak_power=('max_active_power', 'max')
).reset_index()
monthly.to_sql('monthly_summary', conn, if_exists='replace', index=False)

load_time = round(time.time() - start, 2)
print(f'\n[LOAD COMPLETE]')
print(f'  hourly_consumption table: {len(df_hourly):,} rows')
print(f'  monthly_summary table:    {len(monthly):,} rows')
print(f'  Database file: {DB_FILE}')
print(f'  Time taken: {load_time}s')


# ══════════════════════════════════════════════════════════════
# VALIDATE
# Run SQL integrity checks on the loaded data
# ══════════════════════════════════════════════════════════════
print('\n[VALIDATE] Running data integrity checks...')

checks = {}

# Check 1 — Row count matches
db_count = pd.read_sql_query('SELECT COUNT(*) as cnt FROM hourly_consumption', conn).iloc[0]['cnt']
checks['Row count matches'] = db_count == len(df_hourly)

# Check 2 — No nulls in key columns
null_check = pd.read_sql_query('''
    SELECT COUNT(*) as cnt FROM hourly_consumption
    WHERE avg_active_power IS NULL OR datetime IS NULL
''', conn).iloc[0]['cnt']
checks['No nulls in key columns'] = null_check == 0

# Check 3 — All power values are non-negative
neg_check = pd.read_sql_query('''
    SELECT COUNT(*) as cnt FROM hourly_consumption
    WHERE avg_active_power < 0
''', conn).iloc[0]['cnt']
checks['No negative power values'] = neg_check == 0

# Check 4 — Date range is continuous (no huge gaps)
date_check = pd.read_sql_query('''
    SELECT MIN(datetime) as min_dt, MAX(datetime) as max_dt,
           COUNT(DISTINCT substr(datetime,1,7)) as months_covered
    FROM hourly_consumption
''', conn)
checks['Date range valid'] = date_check.iloc[0]['months_covered'] > 12

print('\n  Validation Results:')
all_passed = True
for check, passed in checks.items():
    status = 'PASS' if passed else 'FAIL'
    print(f'  [{status}] {check}')
    if not passed:
        all_passed = False

print(f'\n  Overall: {"ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"}')


# ══════════════════════════════════════════════════════════════
# ANALYSE + VISUALISE
# Run SQL queries and plot key consumption patterns
# ══════════════════════════════════════════════════════════════
print('\n[ANALYSE] Running consumption analysis queries...')

# Query 1 — Average consumption by hour of day
hourly_pattern = pd.read_sql_query('''
    SELECT hour,
           ROUND(AVG(avg_active_power), 3) AS avg_power_kw,
           ROUND(AVG(total_energy_wh) / 1000, 4) AS avg_energy_kwh
    FROM hourly_consumption
    GROUP BY hour
    ORDER BY hour
''', conn)

# Query 2 — Average consumption by day of week
daily_pattern = pd.read_sql_query('''
    SELECT day_name,
           ROUND(AVG(avg_active_power), 3) AS avg_power_kw
    FROM hourly_consumption
    WHERE day_name IS NOT NULL
    GROUP BY day_name
    ORDER BY AVG(avg_active_power) DESC
''', conn)

# Query 3 — Monthly total consumption
monthly_consumption = pd.read_sql_query('''
    SELECT year_month, total_kwh
    FROM monthly_summary
    ORDER BY year_month
''', conn)

# Query 4 — Weekend vs weekday
weekend_pattern = pd.read_sql_query('''
    SELECT
        CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
        ROUND(AVG(avg_active_power), 3) AS avg_power_kw,
        ROUND(AVG(total_energy_wh) / 1000, 4) AS avg_energy_kwh
    FROM hourly_consumption
    GROUP BY is_weekend
''', conn)

print('  Queries complete')

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Smart Meter Energy Consumption Analysis — UCI Household Dataset',
             fontsize=13, fontweight='bold', y=1.01)

# Panel 1 — Avg power by hour of day
axes[0, 0].plot(
    hourly_pattern['hour'],
    hourly_pattern['avg_power_kw'],
    color='#378ADD', linewidth=2, marker='o', markersize=4
)
axes[0, 0].fill_between(hourly_pattern['hour'], hourly_pattern['avg_power_kw'],
                         alpha=0.15, color='#378ADD')
axes[0, 0].set_xlabel('Hour of day')
axes[0, 0].set_ylabel('Avg active power (kW)')
axes[0, 0].set_title('Average power consumption by hour')
axes[0, 0].set_xticks(range(0, 24, 2))

# Panel 2 — Avg power by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_plot = daily_pattern.set_index('day_name').reindex(day_order).reset_index()
bar_colors = ['#E24B4A' if d in ['Saturday', 'Sunday'] else '#B5D4F4'
              for d in daily_plot['day_name']]
axes[0, 1].bar(daily_plot['day_name'], daily_plot['avg_power_kw'], color=bar_colors)
axes[0, 1].set_xlabel('Day of week')
axes[0, 1].set_ylabel('Avg active power (kW)')
axes[0, 1].set_title('Average power by day (red = weekend)')
axes[0, 1].tick_params(axis='x', rotation=30)

# Panel 3 — Monthly consumption
axes[1, 0].bar(
    range(len(monthly_consumption)),
    monthly_consumption['total_kwh'],
    color='#1D9E75'
)
axes[1, 0].set_xticks(range(len(monthly_consumption)))
axes[1, 0].set_xticklabels(
    monthly_consumption['year_month'],
    rotation=45, ha='right', fontsize=7
)
axes[1, 0].set_ylabel('Total consumption (kWh)')
axes[1, 0].set_title('Monthly total energy consumption')

# Panel 4 — Weekend vs weekday
bars = axes[1, 1].bar(
    weekend_pattern['day_type'],
    weekend_pattern['avg_power_kw'],
    color=['#EF9F27', '#378ADD'],
    width=0.4
)
axes[1, 1].set_ylabel('Avg active power (kW)')
axes[1, 1].set_title('Weekday vs weekend consumption')
for bar, val in zip(bars, weekend_pattern['avg_power_kw']):
    axes[1, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f'{val} kW', ha='center', va='bottom',
        fontsize=10, fontweight='bold'
    )

plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150, bbox_inches='tight')
print(f'  Chart saved as {CHART_FILE}')
plt.show()


# ══════════════════════════════════════════════════════════════
# EXPORT + BUSINESS INSIGHT SUMMARY
# ══════════════════════════════════════════════════════════════
hourly_pattern.to_csv(f'{OUTPUT_DIR}/consumption_by_hour.csv', index=False)
daily_pattern.to_csv(f'{OUTPUT_DIR}/consumption_by_day.csv', index=False)
monthly_consumption.to_csv(f'{OUTPUT_DIR}/monthly_consumption.csv', index=False)

conn.close()

peak_hour  = hourly_pattern.loc[hourly_pattern['avg_power_kw'].idxmax(), 'hour']
low_hour   = hourly_pattern.loc[hourly_pattern['avg_power_kw'].idxmin(), 'hour']
peak_month = monthly_consumption.loc[monthly_consumption['total_kwh'].idxmax(), 'year_month']
total_kwh  = monthly_consumption['total_kwh'].sum()

print('\n' + '=' * 60)
print('BUSINESS INSIGHT SUMMARY')
print('=' * 60)
print(f'Total ETL time:          {round(extract_time + transform_time + load_time, 2)}s')
print(f'Raw rows processed:      {len(df_raw):,}')
print(f'Clean hourly records:    {len(df_hourly):,}')
print(f'Months of data:          {len(monthly_consumption)}')
print(f'Total energy consumed:   {total_kwh:,.0f} kWh')
print()
print(f'Peak consumption hour:   {peak_hour}:00 — highest avg demand')
print(f'Lowest consumption hour: {low_hour}:00 — best time for EV charging / appliances')
print(f'Highest consumption month: {peak_month}')
print()
wkday = weekend_pattern[weekend_pattern['day_type'] == 'Weekday']['avg_power_kw'].values[0]
wkend = weekend_pattern[weekend_pattern['day_type'] == 'Weekend']['avg_power_kw'].values[0]
diff  = round(abs(wkend - wkday) / wkday * 100, 1)
higher = 'Weekend' if wkend > wkday else 'Weekday'
print(f'Weekend vs weekday:      {higher} is {diff}% higher in avg consumption')
print()
print('Recommendations for energy analyst:')
print(f'  1. Shift non-essential loads to {low_hour}:00 — lowest demand window')
print(f'  2. {peak_month} is peak month — review tariff and demand response options')
print(f'  3. Pipeline runs end-to-end in {round(extract_time + transform_time + load_time, 2)}s — schedule weekly via cron')
print()
print(f'Output files:')
print(f'  Database:  {DB_FILE}')
print(f'  Chart:     {CHART_FILE}')
print(f'  CSVs:      {OUTPUT_DIR}/')
print('=' * 60)
