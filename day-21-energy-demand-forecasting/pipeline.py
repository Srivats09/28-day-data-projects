"""
pipeline.py
===========
Day 21: Energy Demand Forecasting Pipeline

Industry:  Energy / Utilities
Format:    Python script (.py)
Skills:    ETL, pandas, numpy, time series, forecasting, matplotlib

Who uses this:
    A grid operations analyst planning generation capacity for the
    next 7 days. This pipeline loads real National Grid half-hourly
    demand data, engineers time features, fits a per-period linear
    trend model, forecasts 7 days ahead, and evaluates accuracy.

Data:
    UK National Grid ESO — Historic Demand Data
    Half-hourly settlement period demand (MW)
    Source: data.nationalgrideso.com (live, no login)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 60)
print('ENERGY DEMAND FORECASTING PIPELINE')
print('=' * 60)


# ══════════════════════════════════════════════════════════════
# EXTRACT
# ══════════════════════════════════════════════════════════════
print('\n[EXTRACT] Loading National Grid demand data...')

df_raw = pd.read_csv('demand_data.csv')

print(f'  Raw rows:             {len(df_raw):,}')
print(f'  Columns:              {df_raw.shape[1]}')
print(f'  Date range (raw):     {df_raw["SETTLEMENT_DATE"].min()} to {df_raw["SETTLEMENT_DATE"].max()}')


# ══════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════
print('\n[TRANSFORM] Engineering features...')

df = df_raw[['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND', 'TSD',
             'ENGLAND_WALES_DEMAND', 'EMBEDDED_WIND_GENERATION',
             'EMBEDDED_SOLAR_GENERATION']].copy()

# Parse datetime — each settlement period = 30 mins
df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])
df['datetime'] = df['SETTLEMENT_DATE'] + pd.to_timedelta(
    (df['SETTLEMENT_PERIOD'] - 1) * 30, unit='m'
)

# Drop nulls and obvious outliers
df = df.dropna(subset=['ND', 'TSD'])
df = df[df['ND'] > 5000]   # remove near-zero bad readings
df = df[df['ND'] < 70000]  # remove unrealistic spikes

# Time features
df['hour']       = df['datetime'].dt.hour
df['minute']     = df['datetime'].dt.minute
df['period']     = df['SETTLEMENT_PERIOD']  # 1-48 half-hour slots per day
df['dayofweek']  = df['datetime'].dt.dayofweek   # 0=Mon, 6=Sun
df['month']      = df['datetime'].dt.month
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['date']       = df['datetime'].dt.date

# Rolling 7-day average (smoothed signal)
df = df.sort_values('datetime').reset_index(drop=True)
df['nd_rolling7d'] = df['ND'].rolling(window=7*48, min_periods=1).mean().round(0)

# Daily aggregates
daily = (
    df.groupby('date')
    .agg(
        mean_demand=('ND', 'mean'),
        peak_demand=('ND', 'max'),
        min_demand=('ND', 'min'),
        mean_wind=('EMBEDDED_WIND_GENERATION', 'mean'),
        mean_solar=('EMBEDDED_SOLAR_GENERATION', 'mean'),
        is_weekend=('is_weekend', 'first')
    )
    .round(0)
    .reset_index()
)
daily['date'] = pd.to_datetime(daily['date'])
daily['dayofweek'] = daily['date'].dt.dayofweek
daily['month']     = daily['date'].dt.month

print(f'  Clean rows:           {len(df):,}')
print(f'  Date range:           {df["datetime"].min().strftime("%Y-%m-%d")} to {df["datetime"].max().strftime("%Y-%m-%d")}')
print(f'  Settlement periods:   {df["period"].nunique()} per day (half-hourly)')
print(f'  Daily records:        {len(daily):,} days')
print(f'  Avg national demand:  {df["ND"].mean():,.0f} MW')
print(f'  Peak demand:          {df["ND"].max():,.0f} MW')
print(f'  Min demand:           {df["ND"].min():,.0f} MW')


# ══════════════════════════════════════════════════════════════
# MODEL — Per-period linear trend + day-of-week adjustment
# ══════════════════════════════════════════════════════════════
print('\n[MODEL] Fitting per-period demand model...')

# Strategy: for each of the 48 half-hour periods in a day,
# fit a linear trend over time and capture day-of-week effects.
# Forecast = trend(t) + day_of_week_adjustment

# Day of week average demand (baseline pattern)
dow_avg = df.groupby('dayofweek')['ND'].mean()
overall_avg = df['ND'].mean()
dow_adjustment = (dow_avg - overall_avg).to_dict()

# Per-period trend model
# Use last 90 days of data for trend fitting
cutoff = df['datetime'].max() - pd.Timedelta(days=90)
df_recent = df[df['datetime'] >= cutoff].copy()
df_recent['t'] = (df_recent['datetime'] - df_recent['datetime'].min()).dt.total_seconds() / 3600

period_models = {}
for period in range(1, 49):
    subset = df_recent[df_recent['period'] == period][['t', 'ND', 'dayofweek']].dropna()
    if len(subset) < 10:
        continue
    # Simple linear regression via numpy
    coeffs = np.polyfit(subset['t'], subset['ND'], 1)
    period_models[period] = {
        'slope'    : coeffs[0],
        'intercept': coeffs[1],
        'mean_nd'  : subset['ND'].mean(),
    }

print(f'  Periods modelled:     {len(period_models)}/48')
print(f'  Training window:      last 90 days')
print(f'  Model:                linear trend per period + day-of-week adjustment')


# ══════════════════════════════════════════════════════════════
# FORECAST — Next 7 days
# ══════════════════════════════════════════════════════════════
print('\n[FORECAST] Generating 7-day demand forecast...')

last_dt  = df['datetime'].max()
t_origin = df_recent['datetime'].min()

forecast_rows = []
for day_offset in range(1, 8):
    for period in range(1, 49):
        forecast_dt  = last_dt + pd.Timedelta(days=day_offset) - pd.Timedelta(hours=last_dt.hour, minutes=last_dt.minute) + pd.Timedelta(minutes=(period-1)*30)
        t_hours      = (forecast_dt - t_origin).total_seconds() / 3600
        dow          = forecast_dt.dayofweek
        is_weekend   = 1 if dow in [5, 6] else 0

        if period in period_models:
            m = period_models[period]
            base_forecast = m['slope'] * t_hours + m['intercept']
            # Apply day-of-week adjustment
            dow_adj       = dow_adjustment.get(dow, 0)
            forecast_mw   = base_forecast + dow_adj * 0.3  # dampened adjustment
        else:
            forecast_mw = df['ND'].mean()

        # Clip to realistic range
        forecast_mw = np.clip(forecast_mw, 10000, 65000)

        forecast_rows.append({
            'datetime'   : forecast_dt,
            'date'       : forecast_dt.date(),
            'period'     : period,
            'hour'       : forecast_dt.hour,
            'dayofweek'  : dow,
            'is_weekend' : is_weekend,
            'forecast_mw': round(forecast_mw, 0),
        })

df_forecast = pd.DataFrame(forecast_rows)

# Daily forecast summary
daily_forecast = (
    df_forecast.groupby('date')
    .agg(
        mean_forecast=('forecast_mw', 'mean'),
        peak_forecast=('forecast_mw', 'max'),
        min_forecast=('forecast_mw', 'min'),
    )
    .round(0)
    .reset_index()
)
daily_forecast['date'] = pd.to_datetime(daily_forecast['date'])
daily_forecast['day_name'] = daily_forecast['date'].dt.strftime('%a %d %b')

print(f'  Forecast periods:     {len(df_forecast):,} half-hour slots')
print(f'\n  7-day forecast summary:')
print(f'  {"Date":15s} {"Day":5s} {"Mean MW":>10s} {"Peak MW":>10s} {"Min MW":>10s}')
print(f'  {"-"*55}')
for _, row in daily_forecast.iterrows():
    dow_name = row['date'].strftime('%a')
    print(f'  {str(row["date"].date()):15s} {dow_name:5s} {row["mean_forecast"]:>10,.0f} {row["peak_forecast"]:>10,.0f} {row["min_forecast"]:>10,.0f}')


# ══════════════════════════════════════════════════════════════
# EVALUATE — MAPE on last 7 days (holdout)
# ══════════════════════════════════════════════════════════════
print('\n[EVALUATE] Calculating accuracy on last 7 days...')

holdout_start = last_dt - pd.Timedelta(days=7)
df_holdout    = df[df['datetime'] > holdout_start].copy()
df_holdout['t'] = (df_holdout['datetime'] - t_origin).dt.total_seconds() / 3600

actuals, predictions = [], []
for _, row in df_holdout.iterrows():
    period = int(row['period'])
    if period in period_models:
        m    = period_models[period]
        pred = m['slope'] * row['t'] + m['intercept']
        pred += dow_adjustment.get(row['dayofweek'], 0) * 0.3
        pred  = np.clip(pred, 10000, 65000)
        actuals.append(row['ND'])
        predictions.append(pred)

actuals     = np.array(actuals)
predictions = np.array(predictions)
mape        = np.mean(np.abs((actuals - predictions) / actuals)) * 100
mae         = np.mean(np.abs(actuals - predictions))
rmse        = np.sqrt(np.mean((actuals - predictions) ** 2))

print(f'  Holdout period:       last 7 days')
print(f'  MAPE:                 {mape:.2f}%')
print(f'  MAE:                  {mae:,.0f} MW')
print(f'  RMSE:                 {rmse:,.0f} MW')


# ══════════════════════════════════════════════════════════════
# VISUALISE
# ══════════════════════════════════════════════════════════════
print('\n[VISUALISE] Building dashboard...')

fig = plt.figure(figsize=(18, 13))
fig.suptitle('UK National Grid — Energy Demand Forecasting Pipeline',
             fontsize=14, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# Panel 1 — Recent actual demand + 7-day forecast
ax1 = fig.add_subplot(gs[0, :])  # full width
recent_daily = daily[daily['date'] >= (daily['date'].max() - pd.Timedelta(days=30))]
ax1.plot(recent_daily['date'], recent_daily['mean_demand'],
         color='#378ADD', linewidth=2, label='Actual (30-day history)')
ax1.fill_between(recent_daily['date'],
                 recent_daily['min_demand'], recent_daily['peak_demand'],
                 alpha=0.15, color='#378ADD')
ax1.plot(daily_forecast['date'], daily_forecast['mean_forecast'],
         color='#E24B4A', linewidth=2.5, linestyle='--',
         marker='o', markersize=6, label='Forecast (7 days)')
ax1.fill_between(daily_forecast['date'],
                 daily_forecast['min_forecast'], daily_forecast['peak_forecast'],
                 alpha=0.15, color='#E24B4A')
ax1.axvline(last_dt, color='gray', linestyle=':', linewidth=1.5, label='Forecast start')
ax1.set_ylabel('National Demand (MW)')
ax1.set_title('Actual demand (30 days) + 7-day forecast')
ax1.legend(fontsize=9)

# Panel 2 — Average demand by hour of day
ax2 = fig.add_subplot(gs[1, 0])
hourly_avg = df.groupby('hour')['ND'].mean()
hourly_we  = df[df['is_weekend']==1].groupby('hour')['ND'].mean()
hourly_wd  = df[df['is_weekend']==0].groupby('hour')['ND'].mean()
ax2.plot(hourly_wd.index, hourly_wd.values, color='#378ADD', linewidth=2, label='Weekday')
ax2.plot(hourly_we.index, hourly_we.values, color='#EF9F27', linewidth=2, label='Weekend')
ax2.set_xlabel('Hour of day')
ax2.set_ylabel('Avg demand (MW)')
ax2.set_title('Average demand profile by hour')
ax2.legend(fontsize=9)
ax2.set_xticks(range(0, 24, 3))

# Panel 3 — Day of week demand pattern
ax3 = fig.add_subplot(gs[1, 1])
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_means = [df[df['dayofweek']==d]['ND'].mean() for d in range(7)]
colors_dow = ['#E24B4A' if d in [5, 6] else '#378ADD' for d in range(7)]
bars = ax3.bar(dow_names, dow_means, color=colors_dow)
ax3.set_ylabel('Avg demand (MW)')
ax3.set_title('Average demand by day of week')
ax3.axhline(np.mean(dow_means), color='gray', linestyle='--', linewidth=1, label='Overall avg')
ax3.legend(fontsize=9)
for bar, val in zip(bars, dow_means):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
             f'{val:,.0f}', ha='center', va='bottom', fontsize=8)

plt.savefig('demand_forecast.png', dpi=150, bbox_inches='tight')
print('  Chart saved as demand_forecast.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════
df_forecast.to_csv(f'{OUTPUT_DIR}/7day_forecast.csv', index=False)
daily_forecast.to_csv(f'{OUTPUT_DIR}/7day_daily_summary.csv', index=False)
daily.to_csv(f'{OUTPUT_DIR}/daily_actuals.csv', index=False)

peak_day     = daily_forecast.loc[daily_forecast['peak_forecast'].idxmax()]
low_day      = daily_forecast.loc[daily_forecast['min_forecast'].idxmin()]
weekend_days = daily_forecast[daily_forecast['date'].dt.dayofweek.isin([5,6])]
weekday_days = daily_forecast[~daily_forecast['date'].dt.dayofweek.isin([5,6])]

print('\n' + '=' * 60)
print('BUSINESS INSIGHT SUMMARY')
print('=' * 60)
print(f'Historical data:           {len(df):,} half-hourly readings')
print(f'Date range:                {df["datetime"].min().strftime("%Y-%m-%d")} to {df["datetime"].max().strftime("%Y-%m-%d")}')
print(f'Avg national demand:       {df["ND"].mean():,.0f} MW')
print(f'Peak demand recorded:      {df["ND"].max():,.0f} MW')
print()
print(f'Model accuracy (last 7d):')
print(f'  MAPE:                    {mape:.2f}%')
print(f'  MAE:                     {mae:,.0f} MW')
print(f'  RMSE:                    {rmse:,.0f} MW')
print()
print(f'7-day forecast:')
print(f'  Highest demand day:      {peak_day["day_name"]} ({peak_day["peak_forecast"]:,.0f} MW peak)')
print(f'  Lowest demand day:       {low_day["day_name"]} ({low_day["min_forecast"]:,.0f} MW min)')
if len(weekend_days) > 0 and len(weekday_days) > 0:
    print(f'  Avg weekday demand:      {weekday_days["mean_forecast"].mean():,.0f} MW')
    print(f'  Avg weekend demand:      {weekend_days["mean_forecast"].mean():,.0f} MW')
print()
print(f'Key patterns:')
wd_avg = df[df['is_weekend']==0]['ND'].mean()
we_avg = df[df['is_weekend']==1]['ND'].mean()
print(f'  Weekday vs weekend gap:  {((wd_avg-we_avg)/we_avg*100):.1f}% higher on weekdays')
peak_hour = df.groupby('hour')['ND'].mean().idxmax()
trough_hr = df.groupby('hour')['ND'].mean().idxmin()
print(f'  Peak demand hour:        {peak_hour:02d}:00')
print(f'  Trough demand hour:      {trough_hr:02d}:00')
print('=' * 60)
