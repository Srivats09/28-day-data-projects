"""
etl_pipeline.py
===============
Day 24: Retail Demand Forecasting ETL

Industry:  Retail / E-commerce
Format:    Python script (.py)
Skills:    ETL, pandas, numpy, SQLite, forecasting, matplotlib

Who uses this:
    A supply chain planner deciding stock orders for the next
    4 weeks. This pipeline loads real supermarket transaction data,
    engineers demand features, loads into SQLite, calculates
    rolling averages, forecasts 4-week demand per product line,
    and flags overstock/understock risk.

Data:
    Supermarket Sales Dataset — 3 branches, 6 product lines
    1,000 real transactions across Jan–Mar 2019
    Source: github.com/sushantag9/Supermarket-Sales-Data-Analysis
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
OUTPUT_DIR = 'output'
DB_FILE    = 'retail_demand.db'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 60)
print('RETAIL DEMAND FORECASTING ETL PIPELINE')
print('=' * 60)


# ══════════════════════════════════════════════════════════════
# EXTRACT
# ══════════════════════════════════════════════════════════════
print('\n[EXTRACT] Loading supermarket sales data...')
start = time.time()

df_raw = pd.read_csv('sales.csv')

print(f'  Raw rows:             {len(df_raw):,}')
print(f'  Columns:              {df_raw.shape[1]}')
print(f'  Branches:             {df_raw["Branch"].unique().tolist()}')
print(f'  Product lines:        {df_raw["Product line"].unique().tolist()}')
print(f'  Date range:           {df_raw["Date"].min()} to {df_raw["Date"].max()}')
print(f'  Time: {round(time.time()-start,2)}s')


# ══════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════
print('\n[TRANSFORM] Cleaning and engineering features...')
start = time.time()

df = df_raw.copy()

# Parse dates and times
df['date']       = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df['time']       = pd.to_datetime(df['Time'], format='%H:%M').dt.time
df['hour']       = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
df['dayofweek']  = df['date'].dt.dayofweek
df['week']       = df['date'].dt.isocalendar().week.astype(int)
df['month']      = df['date'].dt.month
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['day_name']   = df['date'].dt.strftime('%A')

# Rename for cleaner SQL
df = df.rename(columns={
    'Invoice ID'    : 'invoice_id',
    'Branch'        : 'branch',
    'City'          : 'city',
    'Customer type' : 'customer_type',
    'Gender'        : 'gender',
    'Product line'  : 'product_line',
    'Unit price'    : 'unit_price',
    'Quantity'      : 'quantity',
    'Tax 5%'        : 'tax',
    'Total'         : 'total_sales',
    'Payment'       : 'payment_method',
    'cogs'          : 'cogs',
    'gross income'  : 'gross_income',
    'Rating'        : 'rating',
})

df = df[['invoice_id','date','branch','city','customer_type','gender',
         'product_line','unit_price','quantity','tax','total_sales',
         'payment_method','cogs','gross_income','rating',
         'hour','dayofweek','week','month','is_weekend','day_name']]

# Validate
assert df['total_sales'].min() > 0, 'Negative sales found'
assert df['quantity'].min() > 0,    'Zero quantity found'
assert df['date'].isna().sum() == 0, 'Null dates found'

# Weekly demand aggregation per product line
weekly_demand = (
    df.groupby(['week', 'product_line'])
    .agg(
        transactions=('invoice_id', 'count'),
        total_qty=('quantity', 'sum'),
        total_revenue=('total_sales', 'sum'),
        avg_unit_price=('unit_price', 'mean'),
        avg_rating=('rating', 'mean'),
    )
    .round(2)
    .reset_index()
    .sort_values(['product_line', 'week'])
)

# Rolling 2-week average per product line (smoothed demand signal)
weekly_demand['rolling2w_qty'] = (
    weekly_demand.groupby('product_line')['total_qty']
    .transform(lambda x: x.rolling(2, min_periods=1).mean())
    .round(1)
)

print(f'  Clean rows:           {len(df):,}')
print(f'  Weeks in data:        {df["week"].nunique()} ({df["week"].min()}–{df["week"].max()})')
print(f'  Total revenue:        £{df["total_sales"].sum():,.2f}')
print(f'  Avg transaction:      £{df["total_sales"].mean():,.2f}')
print(f'  Time: {round(time.time()-start,2)}s')


# ══════════════════════════════════════════════════════════════
# LOAD — SQLite
# ══════════════════════════════════════════════════════════════
print('\n[LOAD] Writing to SQLite...')
start = time.time()

conn = sqlite3.connect(DB_FILE)
df.to_sql('transactions', conn, if_exists='replace', index=False)
weekly_demand.to_sql('weekly_demand', conn, if_exists='replace', index=False)

# SQL summary views
branch_summary = pd.read_sql_query("""
    SELECT
        branch, city,
        COUNT(*)                       AS transactions,
        ROUND(SUM(total_sales), 2)     AS total_revenue,
        ROUND(AVG(total_sales), 2)     AS avg_transaction,
        ROUND(AVG(rating), 2)          AS avg_rating,
        ROUND(SUM(gross_income), 2)    AS total_profit
    FROM transactions
    GROUP BY branch, city
    ORDER BY total_revenue DESC
""", conn)

product_summary = pd.read_sql_query("""
    SELECT
        product_line,
        COUNT(*)                       AS transactions,
        SUM(quantity)                  AS total_qty,
        ROUND(SUM(total_sales), 2)     AS total_revenue,
        ROUND(AVG(unit_price), 2)      AS avg_unit_price,
        ROUND(AVG(rating), 2)          AS avg_rating,
        ROUND(SUM(gross_income), 2)    AS total_profit
    FROM transactions
    GROUP BY product_line
    ORDER BY total_revenue DESC
""", conn)

payment_summary = pd.read_sql_query("""
    SELECT
        payment_method,
        COUNT(*)                   AS transactions,
        ROUND(SUM(total_sales),2)  AS total_revenue,
        ROUND(AVG(total_sales),2)  AS avg_transaction
    FROM transactions
    GROUP BY payment_method
    ORDER BY total_revenue DESC
""", conn)

print(f'  transactions table:   {len(df):,} rows')
print(f'  weekly_demand table:  {len(weekly_demand):,} rows')
print(f'  Time: {round(time.time()-start,2)}s')

print('\n=== Branch performance ==='          )
print(branch_summary.to_string(index=False))
print('\n=== Product line performance ===')
print(product_summary.to_string(index=False))


# ══════════════════════════════════════════════════════════════
# FORECAST — 4-week demand per product line
# Using rolling avg trend extrapolation
# ══════════════════════════════════════════════════════════════
print('\n[FORECAST] Generating 4-week demand forecast...')

last_week   = weekly_demand['week'].max()
product_lines = weekly_demand['product_line'].unique()

forecast_rows = []
for pl in product_lines:
    pl_data = weekly_demand[weekly_demand['product_line'] == pl].sort_values('week')

    # Linear trend on quantity over weeks
    if len(pl_data) >= 3:
        weeks   = pl_data['week'].values
        qty     = pl_data['total_qty'].values
        coeffs  = np.polyfit(weeks, qty, 1)
        slope, intercept = coeffs
    else:
        slope, intercept = 0, pl_data['total_qty'].mean()

    # Last 2-week rolling avg as baseline
    baseline = pl_data['rolling2w_qty'].iloc[-1]

    for w_offset in range(1, 5):
        forecast_week = last_week + w_offset
        trend_forecast = slope * forecast_week + intercept
        # Blend trend (40%) and baseline (60%)
        blended = (0.4 * trend_forecast + 0.6 * baseline)
        blended = max(blended, 0)

        # Stock recommendation
        safety_stock = blended * 0.20  # 20% safety buffer
        order_qty    = round(blended + safety_stock)

        # Compare to best historical week
        hist_max = pl_data['total_qty'].max()
        hist_avg = pl_data['total_qty'].mean()
        flag = 'OVERSTOCK RISK' if blended > hist_max * 1.1 else \
               'UNDERSTOCK RISK' if blended < hist_avg * 0.7 else 'OK'

        forecast_rows.append({
            'product_line'   : pl,
            'forecast_week'  : forecast_week,
            'forecast_qty'   : round(blended, 1),
            'order_qty'      : order_qty,
            'safety_stock'   : round(safety_stock, 1),
            'trend_slope'    : round(slope, 3),
            'stock_flag'     : flag,
        })

df_forecast = pd.DataFrame(forecast_rows)
df_forecast.to_sql('demand_forecast', conn, if_exists='replace', index=False)

print(f'\n  4-week demand forecast by product line:')
pivot = df_forecast.pivot_table(
    index='product_line', columns='forecast_week',
    values='forecast_qty', aggfunc='first'
).round(1)
print(pivot.to_string())

print(f'\n  Stock flags:')
flags = df_forecast[df_forecast['stock_flag'] != 'OK'][['product_line','forecast_week','stock_flag']]
if len(flags):
    print(flags.to_string(index=False))
else:
    print('  No risk flags — all product lines within normal range')

conn.close()


# ══════════════════════════════════════════════════════════════
# VISUALISE
# ══════════════════════════════════════════════════════════════
print('\n[VISUALISE] Building dashboard...')

fig = plt.figure(figsize=(18, 13))
fig.suptitle('Retail Demand Forecasting ETL — Supermarket Sales Analysis',
             fontsize=14, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

colors_pl = ['#378ADD','#1D9E75','#EF9F27','#E24B4A','#534AB7','#888780']
pl_color  = dict(zip(product_lines, colors_pl))

# Panel 1 — Weekly demand per product line (actual + forecast)
ax1 = fig.add_subplot(gs[0, :])
for pl in product_lines:
    hist = weekly_demand[weekly_demand['product_line'] == pl].sort_values('week')
    fore = df_forecast[df_forecast['product_line'] == pl].sort_values('forecast_week')
    color = pl_color.get(pl, '#888')
    ax1.plot(hist['week'], hist['total_qty'], color=color, linewidth=2,
             marker='o', markersize=4, label=pl)
    ax1.plot(fore['forecast_week'], fore['forecast_qty'], color=color,
             linewidth=2, linestyle='--', marker='s', markersize=5)

ax1.axvline(last_week + 0.5, color='gray', linestyle=':', linewidth=1.5,
            label='Forecast start')
ax1.set_xlabel('Week number')
ax1.set_ylabel('Units sold / forecast')
ax1.set_title('Weekly demand per product line — actual (solid) + 4-week forecast (dashed)')
ax1.legend(fontsize=8, ncol=4)

# Panel 2 — Revenue by product line
ax2 = fig.add_subplot(gs[1, 0])
prod_sorted = product_summary.sort_values('total_revenue', ascending=True)
bar_colors  = [pl_color.get(pl, '#888') for pl in prod_sorted['product_line']]
ax2.barh(prod_sorted['product_line'], prod_sorted['total_revenue'], color=bar_colors)
ax2.set_xlabel('Total revenue ($)')
ax2.set_title('Total revenue by product line')
ax2.tick_params(axis='y', labelsize=8)

# Panel 3 — Hour of day demand pattern
ax3 = fig.add_subplot(gs[1, 1])
hourly = df.groupby('hour')['quantity'].sum()
ax3.bar(hourly.index, hourly.values, color='#378ADD', alpha=0.8)
ax3.set_xlabel('Hour of day')
ax3.set_ylabel('Total units sold')
ax3.set_title('Sales volume by hour of day')
ax3.set_xticks(range(10, 21))

plt.savefig('demand_forecast.png', dpi=150, bbox_inches='tight')
print('  Chart saved as demand_forecast.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════
df_forecast.to_csv(f'{OUTPUT_DIR}/4week_demand_forecast.csv', index=False)
product_summary.to_csv(f'{OUTPUT_DIR}/product_line_summary.csv', index=False)
branch_summary.to_csv(f'{OUTPUT_DIR}/branch_summary.csv', index=False)
weekly_demand.to_csv(f'{OUTPUT_DIR}/weekly_demand.csv', index=False)

top_pl     = product_summary.iloc[0]
bottom_pl  = product_summary.iloc[-1]
top_branch = branch_summary.iloc[0]
peak_hour  = df.groupby('hour')['quantity'].sum().idxmax()
top_payment= payment_summary.iloc[0]
n_flags    = len(df_forecast[df_forecast['stock_flag'] != 'OK'])

print('\n' + '=' * 60)
print('BUSINESS INSIGHT SUMMARY')
print('=' * 60)
print(f'Transactions analysed:     {len(df):,}')
print(f'Date range:                {df["date"].min().strftime("%Y-%m-%d")} to {df["date"].max().strftime("%Y-%m-%d")}')
print(f'Total revenue:             ${df["total_sales"].sum():,.2f}')
print(f'Total profit:              ${df["gross_income"].sum():,.2f}')
print(f'Avg transaction value:     ${df["total_sales"].mean():,.2f}')
print()
print(f'TOP PRODUCT LINE:          {top_pl["product_line"]}')
print(f'  Revenue:                 ${top_pl["total_revenue"]:,.2f}')
print(f'  Avg rating:              {top_pl["avg_rating"]}')
print()
print(f'LOWEST PRODUCT LINE:       {bottom_pl["product_line"]}')
print(f'  Revenue:                 ${bottom_pl["total_revenue"]:,.2f}')
print()
print(f'TOP BRANCH:                {top_branch["branch"]} — {top_branch["city"]}')
print(f'  Revenue:                 ${top_branch["total_revenue"]:,.2f}')
print(f'  Avg rating:              {top_branch["avg_rating"]}')
print()
print(f'PEAK SALES HOUR:           {peak_hour:02d}:00')
print(f'TOP PAYMENT METHOD:        {top_payment["payment_method"]} ({top_payment["transactions"]} transactions)')
print()
print(f'4-WEEK FORECAST:')
print(f'  Stock flags raised:      {n_flags}')
for pl in product_lines:
    pl_fore = df_forecast[df_forecast['product_line']==pl]
    avg_fore = pl_fore['forecast_qty'].mean()
    print(f'  {pl:30s}: avg {avg_fore:.0f} units/week forecast')
print('=' * 60)
