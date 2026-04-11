"""
etl_pipeline.py
===============
Day 17: Logistics Carbon Footprint ETL

Industry:  Transport / Logistics
Format:    Python script (.py)
Skills:    ETL, pandas, sqlite3, matplotlib, carbon accounting

Who uses this:
    A sustainability manager building a quarterly ESG carbon report.
    Instead of manually calculating CO2 per shipment in spreadsheets,
    this pipeline processes the full shipment ledger and outputs
    a ranked emissions report with mode-swap recommendations.

Data:
    Emissions factors: UK Government BEIS 2023 GHG Conversion Factors
    (embedded — kg CO2e per tonne-km by transport mode)
    Shipments: Synthetic — mirrors real freight management system schema
    (Origin, Destination, Weight_kg, Distance_km, Mode, Date)
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = 'output'
DB_FILE    = 'carbon_log.db'
CHART_FILE = 'carbon_footprint_analysis.png'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 60)
print('LOGISTICS CARBON FOOTPRINT ETL PIPELINE')
print('=' * 60)

# ══════════════════════════════════════════════════════════════
# EMISSIONS FACTORS TABLE
# Source: UK Government BEIS GHG Conversion Factors 2023
# Unit: kg CO2e per tonne-km
# ══════════════════════════════════════════════════════════════
EMISSIONS_FACTORS = {
    'Road (HGV)'         : 0.0626,
    'Road (Van)'         : 0.2073,
    'Rail (Freight)'     : 0.0280,
    'Air (Freight)'      : 0.8020,
    'Sea (Container)'    : 0.0113,
    'Sea (Bulk)'         : 0.0078,
}

# Mode alternatives for recommendations
MODE_ALTERNATIVES = {
    'Air (Freight)'  : ('Sea (Container)', 'Road (HGV)'),
    'Road (Van)'     : ('Road (HGV)', 'Rail (Freight)'),
    'Road (HGV)'     : ('Rail (Freight)', 'Sea (Container)'),
    'Sea (Container)': ('Rail (Freight)',),
    'Sea (Bulk)'     : ('Rail (Freight)',),
    'Rail (Freight)' : ('Sea (Container)',),
}

ef_df = pd.DataFrame([
    {'mode': mode, 'kg_co2e_per_tonne_km': factor}
    for mode, factor in EMISSIONS_FACTORS.items()
])

print(f'\n[FACTORS] BEIS 2023 Emissions Factors loaded:')
print(ef_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════
# EXTRACT — Generate synthetic shipment data
# Schema mirrors real freight management system exports
# ══════════════════════════════════════════════════════════════
print('\n[EXTRACT] Generating shipment data...')
start = time.time()

ROUTES = [
    ('London',      'Manchester',  320,   'Road (HGV)'),
    ('London',      'Edinburgh',   650,   'Road (HGV)'),
    ('Birmingham',  'Glasgow',     480,   'Rail (Freight)'),
    ('London',      'New York',    5570,  'Air (Freight)'),
    ('London',      'Dubai',       5500,  'Air (Freight)'),
    ('Southampton', 'Rotterdam',   310,   'Sea (Container)'),
    ('London',      'Singapore',   10840, 'Sea (Container)'),
    ('Manchester',  'Berlin',      1100,  'Road (HGV)'),
    ('Bristol',     'Paris',       390,   'Road (Van)'),
    ('Liverpool',   'Hamburg',     970,   'Sea (Container)'),
    ('London',      'Mumbai',      7200,  'Air (Freight)'),
    ('Birmingham',  'Bristol',     150,   'Road (Van)'),
    ('London',      'Sydney',      16990, 'Air (Freight)'),
    ('Southampton', 'Shanghai',    19600, 'Sea (Bulk)'),
    ('Glasgow',     'Dublin',      290,   'Sea (Container)'),
]

dates = pd.date_range('2024-01-01', '2024-12-31', freq='W')
rows = []
shipment_id = 1

for date in dates:
    n_shipments = np.random.randint(8, 20)
    for _ in range(n_shipments):
        origin, dest, distance, mode = ROUTES[np.random.randint(0, len(ROUTES))]
        weight_kg = np.random.choice([500, 1000, 2000, 5000, 10000, 20000],
                                      p=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05])
        unit_value = np.random.uniform(500, 50000)
        rows.append({
            'shipment_id'   : f'SHP-{shipment_id:05d}',
            'date'          : date.strftime('%Y-%m-%d'),
            'origin'        : origin,
            'destination'   : dest,
            'weight_kg'     : weight_kg,
            'distance_km'   : distance,
            'mode'          : mode,
            'cargo_value_gbp': round(unit_value, 2),
        })
        shipment_id += 1

df_shipments = pd.DataFrame(rows)
print(f'  Shipments generated:  {len(df_shipments):,}')
print(f'  Date range:           {df_shipments["date"].min()} to {df_shipments["date"].max()}')
print(f'  Modes:                {df_shipments["mode"].unique().tolist()}')
print(f'  Time taken:           {round(time.time()-start, 2)}s')


# ══════════════════════════════════════════════════════════════
# TRANSFORM — Calculate CO2e per shipment
# Formula: CO2e_kg = (weight_kg / 1000) * distance_km * factor
# ══════════════════════════════════════════════════════════════
print('\n[TRANSFORM] Calculating CO2 emissions...')
start = time.time()

df = df_shipments.merge(ef_df, on='mode', how='left')

df['weight_tonnes']  = df['weight_kg'] / 1000
df['tonne_km']       = df['weight_tonnes'] * df['distance_km']
df['co2e_kg']        = (df['tonne_km'] * df['kg_co2e_per_tonne_km']).round(2)
df['co2e_tonnes']    = (df['co2e_kg'] / 1000).round(4)

df['date']           = pd.to_datetime(df['date'])
df['month']          = df['date'].dt.to_period('M').astype(str)
df['quarter']        = 'Q' + df['date'].dt.quarter.astype(str)
df['route']          = df['origin'] + ' → ' + df['destination']

# Best alternative mode for each shipment
def best_alternative(mode):
    alts = MODE_ALTERNATIVES.get(mode, ())
    if not alts:
        return mode
    return min(alts, key=lambda m: EMISSIONS_FACTORS.get(m, 999))

df['best_alt_mode']  = df['mode'].apply(best_alternative)
df['alt_factor']     = df['best_alt_mode'].map(EMISSIONS_FACTORS)
df['alt_co2e_kg']    = (df['tonne_km'] * df['alt_factor']).round(2)
df['co2e_saving_kg'] = (df['co2e_kg'] - df['alt_co2e_kg']).clip(lower=0).round(2)

print(f'  Rows processed:       {len(df):,}')
print(f'  Total CO2e:           {df["co2e_tonnes"].sum():,.1f} tonnes')
print(f'  Potential savings:    {df["co2e_saving_kg"].sum()/1000:,.1f} tonnes CO2e')
print(f'  Time taken:           {round(time.time()-start, 2)}s')


# ══════════════════════════════════════════════════════════════
# LOAD — Insert into SQLite carbon_log table
# ══════════════════════════════════════════════════════════════
print('\n[LOAD] Writing to SQLite...')
start = time.time()

conn = sqlite3.connect(DB_FILE)

df.to_sql('carbon_log', conn, if_exists='replace', index=False)
ef_df.to_sql('emissions_factors', conn, if_exists='replace', index=False)

# Monthly summary table
monthly = (
    df.groupby(['month', 'mode'])
    .agg(
        shipments=('shipment_id', 'count'),
        total_co2e_tonnes=('co2e_tonnes', 'sum'),
        total_tonne_km=('tonne_km', 'sum')
    )
    .round(2)
    .reset_index()
)
monthly.to_sql('monthly_summary', conn, if_exists='replace', index=False)

print(f'  carbon_log:           {len(df):,} rows')
print(f'  emissions_factors:    {len(ef_df)} rows')
print(f'  monthly_summary:      {len(monthly)} rows')
print(f'  Time taken:           {round(time.time()-start, 2)}s')


# ══════════════════════════════════════════════════════════════
# VALIDATE
# ══════════════════════════════════════════════════════════════
print('\n[VALIDATE] Running integrity checks...')

checks = {}

row_count = pd.read_sql_query('SELECT COUNT(*) as c FROM carbon_log', conn).iloc[0]['c']
checks['Row count matches']         = int(row_count) == len(df)

null_co2 = pd.read_sql_query('SELECT COUNT(*) as c FROM carbon_log WHERE co2e_kg IS NULL', conn).iloc[0]['c']
checks['No null CO2 values']        = int(null_co2) == 0

neg_co2 = pd.read_sql_query('SELECT COUNT(*) as c FROM carbon_log WHERE co2e_kg < 0', conn).iloc[0]['c']
checks['All CO2 values positive']   = int(neg_co2) == 0

modes_count = pd.read_sql_query('SELECT COUNT(DISTINCT mode) as c FROM carbon_log', conn).iloc[0]['c']
checks['All 6 modes present']       = int(modes_count) == 6

for check, passed in checks.items():
    print(f'  [{"PASS" if passed else "FAIL"}] {check}')

print(f'\n  Overall: {"ALL CHECKS PASSED" if all(checks.values()) else "SOME CHECKS FAILED"}')


# ══════════════════════════════════════════════════════════════
# ANALYSE
# ══════════════════════════════════════════════════════════════
print('\n[ANALYSE] Running analysis queries...')

# Top 10 highest emission routes
top_routes = (
    df.groupby('route')
    .agg(
        shipments=('shipment_id', 'count'),
        total_co2e=('co2e_tonnes', 'sum'),
        avg_co2e=('co2e_tonnes', 'mean'),
        total_saving=('co2e_saving_kg', 'sum')
    )
    .round(2)
    .sort_values('total_co2e', ascending=False)
    .head(10)
    .reset_index()
)

# CO2 by mode
by_mode = (
    df.groupby('mode')
    .agg(
        shipments=('shipment_id', 'count'),
        total_co2e=('co2e_tonnes', 'sum'),
        avg_co2e_per_shipment=('co2e_tonnes', 'mean'),
        pct_of_total=('co2e_tonnes', lambda x: x.sum() / df['co2e_tonnes'].sum() * 100)
    )
    .round(2)
    .sort_values('total_co2e', ascending=False)
    .reset_index()
)

# Monthly trend
monthly_trend = (
    df.groupby('month')['co2e_tonnes']
    .sum()
    .round(2)
    .reset_index()
    .sort_values('month')
)

# Top saving opportunities
top_savings = (
    df.groupby('route')
    .agg(
        total_saving_tonnes=('co2e_saving_kg', lambda x: x.sum() / 1000),
        mode=('mode', 'first'),
        best_alt=('best_alt_mode', 'first')
    )
    .round(2)
    .sort_values('total_saving_tonnes', ascending=False)
    .head(5)
    .reset_index()
)

conn.close()

print('  Queries complete')
print('\n=== Top 10 emission routes ===')
print(top_routes[['route','shipments','total_co2e','total_saving']].to_string())
print('\n=== CO2 by transport mode ===')
print(by_mode.to_string())


# ══════════════════════════════════════════════════════════════
# VISUALISE
# ══════════════════════════════════════════════════════════════
print('\n[VISUALISE] Building dashboard...')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Logistics Carbon Footprint Analysis — 2024 Shipments',
             fontsize=14, fontweight='bold', y=1.01)

# Panel 1 — Top 10 emission routes
axes[0,0].barh(top_routes['route'], top_routes['total_co2e'], color='#E24B4A')
axes[0,0].set_xlabel('Total CO2e (tonnes)')
axes[0,0].set_title('Top 10 highest-emission routes')
axes[0,0].invert_yaxis()
axes[0,0].tick_params(axis='y', labelsize=8)

# Panel 2 — CO2 by transport mode
mode_colors = {
    'Air (Freight)'  : '#E24B4A',
    'Road (Van)'     : '#EF9F27',
    'Road (HGV)'     : '#B5D4F4',
    'Rail (Freight)' : '#1D9E75',
    'Sea (Container)': '#378ADD',
    'Sea (Bulk)'     : '#085041',
}
bar_colors = [mode_colors.get(m, '#888') for m in by_mode['mode']]
bars = axes[0,1].bar(by_mode['mode'], by_mode['total_co2e'], color=bar_colors)
axes[0,1].set_ylabel('Total CO2e (tonnes)')
axes[0,1].set_title('Total emissions by transport mode')
axes[0,1].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, by_mode['pct_of_total']):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

# Panel 3 — Monthly CO2 trend
axes[1,0].plot(range(len(monthly_trend)), monthly_trend['co2e_tonnes'],
               color='#378ADD', linewidth=2.5, marker='o', markersize=5)
axes[1,0].fill_between(range(len(monthly_trend)), monthly_trend['co2e_tonnes'],
                        alpha=0.1, color='#378ADD')
axes[1,0].set_xticks(range(len(monthly_trend)))
axes[1,0].set_xticklabels(monthly_trend['month'], rotation=30, fontsize=7)
axes[1,0].set_ylabel('CO2e (tonnes)')
axes[1,0].set_title('Monthly emissions trend')

# Panel 4 — Savings opportunity by route
axes[1,1].barh(top_savings['route'],
               top_savings['total_saving_tonnes'], color='#1D9E75')
axes[1,1].set_xlabel('Potential CO2e saving (tonnes)')
axes[1,1].set_title('Top 5 mode-swap saving opportunities')
axes[1,1].invert_yaxis()
axes[1,1].tick_params(axis='y', labelsize=8)
for i, row in top_savings.iterrows():
    axes[1,1].text(0.1, list(top_savings.index).index(i),
                   f'  → {row["best_alt"]}', va='center', fontsize=7,
                   color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150, bbox_inches='tight')
print(f'  Chart saved as {CHART_FILE}')
plt.show()


# ══════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════
top_routes.to_csv(f'{OUTPUT_DIR}/top_emission_routes.csv', index=False)
by_mode.to_csv(f'{OUTPUT_DIR}/emissions_by_mode.csv', index=False)
top_savings.to_csv(f'{OUTPUT_DIR}/saving_opportunities.csv', index=False)
monthly_trend.to_csv(f'{OUTPUT_DIR}/monthly_trend.csv', index=False)

total_co2      = df['co2e_tonnes'].sum()
total_saving   = df['co2e_saving_kg'].sum() / 1000
saving_pct     = total_saving / total_co2 * 100
worst_route    = top_routes.iloc[0]
worst_mode     = by_mode.iloc[0]
best_mode      = by_mode.iloc[-1]

print('\n' + '=' * 60)
print('BUSINESS INSIGHT SUMMARY')
print('=' * 60)
print(f'Total shipments:           {len(df):,}')
print(f'Total CO2e emitted:        {total_co2:,.1f} tonnes')
print(f'Total tonne-km:            {df["tonne_km"].sum():,.0f}')
print()
print(f'Highest emission mode:     {worst_mode["mode"]} ({worst_mode["pct_of_total"]:.1f}% of total)')
print(f'Lowest emission mode:      {best_mode["mode"]} ({best_mode["pct_of_total"]:.1f}% of total)')
print()
print(f'Highest emission route:    {worst_route["route"]}')
print(f'  Total CO2e:              {worst_route["total_co2e"]:,.1f} tonnes')
print(f'  Shipments:               {worst_route["shipments"]}')
print()
print(f'Potential CO2e saving:     {total_saving:,.1f} tonnes ({saving_pct:.1f}% reduction)')
print()
print('TOP MODE-SWAP RECOMMENDATIONS:')
for _, row in top_savings.iterrows():
    print(f'  {row["route"][:40]:40s}')
    print(f'    Switch {row["mode"]} → {row["best_alt"]}')
    print(f'    Saving: {row["total_saving_tonnes"]:.1f} tonnes CO2e')
print('=' * 60)
