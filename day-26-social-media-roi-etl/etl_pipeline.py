"""
etl_pipeline.py
===============
Day 26: Social Media ROI ETL Pipeline

Industry:  Marketing / Digital Media
Format:    Python script (.py)
Skills:    ETL, pandas, numpy, SQLite, matplotlib, marketing analytics

Who uses this:
    A digital marketing manager presenting the monthly social media
    performance report to the CMO. This pipeline processes campaign
    data across 5 platforms, calculates ROAS, CPM, CTR, and engagement
    rate, loads into SQLite, and outputs a ranked ROI report.

Data:
    Synthetic — mirrors real social media ad platform export schema
    (Meta Ads Manager, LinkedIn Campaign Manager, Twitter/X Ads,
    TikTok Ads, Google Display). 6 months, 5 platforms, 8 industries.
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
DB_FILE    = 'social_roi.db'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 60)
print('SOCIAL MEDIA ROI ETL PIPELINE')
print('=' * 60)


# ══════════════════════════════════════════════════════════════
# EXTRACT — Generate realistic social media campaign data
# ══════════════════════════════════════════════════════════════
print('\n[EXTRACT] Generating social media campaign data...')
start = time.time()

PLATFORMS = ['Meta (Facebook/Instagram)', 'LinkedIn', 'Twitter/X', 'TikTok', 'Google Display']
INDUSTRIES = ['E-commerce', 'SaaS', 'Healthcare', 'Finance', 'Education', 'Retail', 'Travel', 'FMCG']
CAMPAIGN_TYPES = ['Brand Awareness', 'Lead Generation', 'Retargeting', 'Conversions', 'Engagement']
OBJECTIVES = ['Reach', 'Traffic', 'Leads', 'Sales', 'Engagement']

# Platform characteristics (realistic benchmarks from industry reports)
PLATFORM_PROFILES = {
    'Meta (Facebook/Instagram)': {
        'cpm_range'     : (6, 18),
        'ctr_range'     : (0.008, 0.025),
        'conv_rate_range': (0.015, 0.045),
        'avg_order_range': (45, 180),
        'budget_range'  : (2000, 25000),
    },
    'LinkedIn': {
        'cpm_range'     : (25, 65),
        'ctr_range'     : (0.004, 0.015),
        'conv_rate_range': (0.008, 0.025),
        'avg_order_range': (200, 800),
        'budget_range'  : (3000, 30000),
    },
    'Twitter/X': {
        'cpm_range'     : (4, 12),
        'ctr_range'     : (0.005, 0.018),
        'conv_rate_range': (0.005, 0.018),
        'avg_order_range': (30, 120),
        'budget_range'  : (1000, 15000),
    },
    'TikTok': {
        'cpm_range'     : (8, 22),
        'ctr_range'     : (0.010, 0.035),
        'conv_rate_range': (0.012, 0.038),
        'avg_order_range': (35, 150),
        'budget_range'  : (2000, 20000),
    },
    'Google Display': {
        'cpm_range'     : (3, 10),
        'ctr_range'     : (0.003, 0.012),
        'conv_rate_range': (0.010, 0.030),
        'avg_order_range': (60, 250),
        'budget_range'  : (2500, 35000),
    },
}

rows = []
campaign_id = 1
dates = pd.date_range('2024-01-01', '2024-06-30', freq='W')

for date in dates:
    for platform in PLATFORMS:
        # 2-5 campaigns per platform per week
        n_campaigns = np.random.randint(2, 6)
        profile = PLATFORM_PROFILES[platform]

        for _ in range(n_campaigns):
            industry      = np.random.choice(INDUSTRIES)
            campaign_type = np.random.choice(CAMPAIGN_TYPES)
            budget        = np.random.uniform(*profile['budget_range'])
            spend         = budget * np.random.uniform(0.75, 1.0)  # actual spend vs budget

            # Impressions from CPM
            cpm           = np.random.uniform(*profile['cpm_range'])
            impressions   = int((spend / cpm) * 1000)

            # Clicks from CTR
            ctr           = np.random.uniform(*profile['ctr_range'])
            clicks        = int(impressions * ctr)

            # Conversions from conversion rate
            conv_rate     = np.random.uniform(*profile['conv_rate_range'])
            conversions   = int(clicks * conv_rate)

            # Revenue from avg order value
            avg_order     = np.random.uniform(*profile['avg_order_range'])
            revenue       = conversions * avg_order

            # Engagement (likes, shares, comments) — higher for TikTok/Meta
            eng_multiplier = 1.5 if platform in ['TikTok', 'Meta (Facebook/Instagram)'] else 1.0
            engagements   = int(impressions * np.random.uniform(0.02, 0.08) * eng_multiplier)

            rows.append({
                'campaign_id'   : f'CMP-{campaign_id:05d}',
                'date'          : date.strftime('%Y-%m-%d'),
                'platform'      : platform,
                'industry'      : industry,
                'campaign_type' : campaign_type,
                'budget'        : round(budget, 2),
                'spend'         : round(spend, 2),
                'impressions'   : impressions,
                'clicks'        : clicks,
                'conversions'   : conversions,
                'revenue'       : round(revenue, 2),
                'engagements'   : engagements,
                'cpm'           : round(cpm, 2),
            })
            campaign_id += 1

df_raw = pd.DataFrame(rows)
print(f'  Campaigns generated:  {len(df_raw):,}')
print(f'  Date range:           {df_raw["date"].min()} to {df_raw["date"].max()}')
print(f'  Platforms:            {df_raw["platform"].nunique()}')
print(f'  Industries:           {df_raw["industry"].nunique()}')
print(f'  Time: {round(time.time()-start, 2)}s')


# ══════════════════════════════════════════════════════════════
# TRANSFORM — Calculate marketing KPIs
# ══════════════════════════════════════════════════════════════
print('\n[TRANSFORM] Calculating marketing KPIs...')
start = time.time()

df = df_raw.copy()
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M').astype(str)
df['week']  = df['date'].dt.isocalendar().week.astype(int)

# Core KPI calculations
df['ctr_pct']          = (df['clicks'] / df['impressions'] * 100).round(3)
df['conv_rate_pct']    = (df['conversions'] / df['clicks'].replace(0, np.nan) * 100).round(3)
df['roas']             = (df['revenue'] / df['spend'].replace(0, np.nan)).round(3)
df['cpa']              = (df['spend'] / df['conversions'].replace(0, np.nan)).round(2)
df['cpc']              = (df['spend'] / df['clicks'].replace(0, np.nan)).round(3)
df['engagement_rate']  = (df['engagements'] / df['impressions'] * 100).round(3)
df['revenue_per_click']= (df['revenue'] / df['clicks'].replace(0, np.nan)).round(3)
df['budget_utilisation']= (df['spend'] / df['budget'] * 100).round(1)
df['profit']           = (df['revenue'] - df['spend']).round(2)
df['roi_pct']          = (df['profit'] / df['spend'] * 100).round(1)

# ROI tier
def roi_tier(roas):
    if roas >= 5:   return 'Excellent (5x+)'
    elif roas >= 3: return 'Good (3-5x)'
    elif roas >= 1: return 'Break-even (1-3x)'
    else:           return 'Loss (<1x)'

df['roi_tier'] = df['roas'].apply(roi_tier)

print(f'  Rows processed:       {len(df):,}')
print(f'  Total spend:          ${df["spend"].sum():,.0f}')
print(f'  Total revenue:        ${df["revenue"].sum():,.0f}')
print(f'  Overall ROAS:         {df["revenue"].sum()/df["spend"].sum():.2f}x')
print(f'  Avg CTR:              {df["ctr_pct"].mean():.3f}%')
print(f'  Avg conv rate:        {df["conv_rate_pct"].mean():.3f}%')
print(f'  Time: {round(time.time()-start, 2)}s')


# ══════════════════════════════════════════════════════════════
# LOAD — SQLite
# ══════════════════════════════════════════════════════════════
print('\n[LOAD] Writing to SQLite...')
start = time.time()

conn = sqlite3.connect(DB_FILE)
df.to_sql('campaigns', conn, if_exists='replace', index=False)

# Platform summary
platform_summary = pd.read_sql_query("""
    SELECT
        platform,
        COUNT(*)                        AS campaigns,
        ROUND(SUM(spend), 0)            AS total_spend,
        ROUND(SUM(revenue), 0)          AS total_revenue,
        ROUND(SUM(revenue)/SUM(spend),3) AS roas,
        ROUND(AVG(ctr_pct), 3)          AS avg_ctr,
        ROUND(AVG(conv_rate_pct), 3)    AS avg_conv_rate,
        ROUND(SUM(impressions)/1e6, 2)  AS impressions_m,
        ROUND(AVG(cpa), 2)              AS avg_cpa,
        ROUND(SUM(profit), 0)           AS total_profit
    FROM campaigns
    GROUP BY platform
    ORDER BY roas DESC
""", conn)

# Monthly trend
monthly_trend = pd.read_sql_query("""
    SELECT
        month,
        ROUND(SUM(spend), 0)            AS total_spend,
        ROUND(SUM(revenue), 0)          AS total_revenue,
        ROUND(SUM(revenue)/SUM(spend),3) AS roas,
        ROUND(AVG(ctr_pct), 3)          AS avg_ctr,
        COUNT(*)                        AS campaigns
    FROM campaigns
    GROUP BY month
    ORDER BY month
""", conn)

# Industry performance
industry_perf = pd.read_sql_query("""
    SELECT
        industry,
        COUNT(*)                         AS campaigns,
        ROUND(SUM(spend), 0)             AS total_spend,
        ROUND(SUM(revenue), 0)           AS total_revenue,
        ROUND(SUM(revenue)/SUM(spend),3)  AS roas,
        ROUND(AVG(cpa), 2)               AS avg_cpa
    FROM campaigns
    GROUP BY industry
    ORDER BY roas DESC
""", conn)

# Campaign type performance
camptype_perf = pd.read_sql_query("""
    SELECT
        campaign_type,
        COUNT(*)                         AS campaigns,
        ROUND(SUM(spend), 0)             AS total_spend,
        ROUND(SUM(revenue)/SUM(spend),3)  AS roas,
        ROUND(AVG(ctr_pct), 3)           AS avg_ctr,
        ROUND(AVG(cpa), 2)               AS avg_cpa
    FROM campaigns
    GROUP BY campaign_type
    ORDER BY roas DESC
""", conn)

conn.close()

print(f'  campaigns table:      {len(df):,} rows')
print(f'  Time: {round(time.time()-start, 2)}s')
print('\n=== Platform ROI Summary ===')
print(platform_summary.to_string(index=False))
print('\n=== Campaign Type Performance ===')
print(camptype_perf.to_string(index=False))


# ══════════════════════════════════════════════════════════════
# VISUALISE
# ══════════════════════════════════════════════════════════════
print('\n[VISUALISE] Building dashboard...')

fig = plt.figure(figsize=(18, 13))
fig.suptitle('Social Media ROI ETL Pipeline — 6-Month Campaign Analysis',
             fontsize=14, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

platform_colors = {
    'Meta (Facebook/Instagram)': '#378ADD',
    'LinkedIn'                 : '#0A66C2',
    'Twitter/X'                : '#1DA1F2',
    'TikTok'                   : '#E24B4A',
    'Google Display'           : '#1D9E75',
}
colors_list = [platform_colors.get(p, '#888') for p in platform_summary['platform']]

# Panel 1 — ROAS by platform
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(platform_summary['platform'], platform_summary['roas'], color=colors_list)
ax1.axhline(1, color='gray', linestyle='--', linewidth=1, label='Break-even (1x)')
ax1.axhline(df['revenue'].sum()/df['spend'].sum(), color='black',
            linestyle=':', linewidth=1.5, label=f'Overall ROAS')
ax1.set_ylabel('ROAS (Return on Ad Spend)')
ax1.set_title('ROAS by platform')
ax1.tick_params(axis='x', rotation=15, labelsize=8)
ax1.legend(fontsize=8)
for bar, val in zip(bars, platform_summary['roas']):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f'{val:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 2 — Monthly ROAS trend
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(range(len(monthly_trend)), monthly_trend['roas'],
         color='#378ADD', linewidth=2.5, marker='o', markersize=7)
ax2.fill_between(range(len(monthly_trend)), monthly_trend['roas'],
                 alpha=0.1, color='#378ADD')
ax2.set_xticks(range(len(monthly_trend)))
ax2.set_xticklabels(monthly_trend['month'], rotation=20, fontsize=9)
ax2.set_ylabel('ROAS')
ax2.set_title('Monthly ROAS trend')
ax2.axhline(1, color='gray', linestyle='--', linewidth=1)

# Panel 3 — Spend vs Revenue by platform (grouped bar)
ax3 = fig.add_subplot(gs[1, 0])
x = range(len(platform_summary))
w = 0.35
ax3.bar([i - w/2 for i in x], platform_summary['total_spend']/1000,
        w, label='Total spend ($000s)', color='#B5D4F4')
ax3.bar([i + w/2 for i in x], platform_summary['total_revenue']/1000,
        w, label='Total revenue ($000s)', color='#1D9E75')
ax3.set_xticks(x)
ax3.set_xticklabels(platform_summary['platform'], rotation=15, ha='right', fontsize=8)
ax3.set_ylabel('Amount ($000s)')
ax3.set_title('Total spend vs revenue by platform')
ax3.legend(fontsize=9)

# Panel 4 — Industry ROAS heatmap-style bar
ax4 = fig.add_subplot(gs[1, 1])
ind_sorted = industry_perf.sort_values('roas', ascending=True)
ind_colors = ['#1D9E75' if r >= 5 else '#EF9F27' if r >= 3 else '#E24B4A'
              for r in ind_sorted['roas']]
ax4.barh(ind_sorted['industry'], ind_sorted['roas'], color=ind_colors)
ax4.axvline(1, color='gray', linestyle='--', linewidth=1)
ax4.set_xlabel('ROAS')
ax4.set_title('ROAS by industry')
for i, (val, cpa) in enumerate(zip(ind_sorted['roas'], ind_sorted['avg_cpa'])):
    ax4.text(val + 0.05, i, f'{val:.2f}x | CPA ${cpa:.0f}',
             va='center', fontsize=8)

plt.savefig('social_roi_dashboard.png', dpi=150, bbox_inches='tight')
print('  Chart saved as social_roi_dashboard.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════
platform_summary.to_csv(f'{OUTPUT_DIR}/platform_roi_summary.csv', index=False)
industry_perf.to_csv(f'{OUTPUT_DIR}/industry_performance.csv', index=False)
camptype_perf.to_csv(f'{OUTPUT_DIR}/campaign_type_performance.csv', index=False)
monthly_trend.to_csv(f'{OUTPUT_DIR}/monthly_trend.csv', index=False)

# Budget reallocation recommendation
top_platform    = platform_summary.iloc[0]
worst_platform  = platform_summary.iloc[-1]
top_industry    = industry_perf.iloc[0]
top_camptype    = camptype_perf.iloc[0]
worst_camptype  = camptype_perf.iloc[-1]
overall_roas    = df['revenue'].sum() / df['spend'].sum()
total_spend     = df['spend'].sum()
total_revenue   = df['revenue'].sum()
total_profit    = df['profit'].sum()
best_month      = monthly_trend.loc[monthly_trend['roas'].idxmax(), 'month']
worst_month     = monthly_trend.loc[monthly_trend['roas'].idxmin(), 'month']

print('\n' + '=' * 60)
print('BUSINESS INSIGHT SUMMARY')
print('=' * 60)
print(f'Campaigns analysed:        {len(df):,}')
print(f'Date range:                {df["date"].min().strftime("%Y-%m-%d")} to {df["date"].max().strftime("%Y-%m-%d")}')
print(f'Total spend:               ${total_spend:,.0f}')
print(f'Total revenue:             ${total_revenue:,.0f}')
print(f'Total profit:              ${total_profit:,.0f}')
print(f'Overall ROAS:              {overall_roas:.2f}x')
print()
print(f'PLATFORM RANKINGS (by ROAS):')
for _, row in platform_summary.iterrows():
    print(f'  {row["platform"]:35s}: {row["roas"]:.2f}x ROAS | CTR {row["avg_ctr"]}% | CPA ${row["avg_cpa"]}')
print()
print(f'BEST PLATFORM:             {top_platform["platform"]} ({top_platform["roas"]:.2f}x ROAS)')
print(f'WORST PLATFORM:            {worst_platform["platform"]} ({worst_platform["roas"]:.2f}x ROAS)')
print()
print(f'BEST CAMPAIGN TYPE:        {top_camptype["campaign_type"]} ({top_camptype["roas"]:.2f}x ROAS)')
print(f'WORST CAMPAIGN TYPE:       {worst_camptype["campaign_type"]} ({worst_camptype["roas"]:.2f}x ROAS)')
print()
print(f'BEST INDUSTRY:             {top_industry["industry"]} ({top_industry["roas"]:.2f}x ROAS)')
print()
print(f'BEST MONTH:                {best_month} (ROAS {monthly_trend.loc[monthly_trend["roas"].idxmax(),"roas"]:.2f}x)')
print(f'WORST MONTH:               {worst_month} (ROAS {monthly_trend.loc[monthly_trend["roas"].idxmin(),"roas"]:.2f}x)')
print()
print('BUDGET REALLOCATION RECOMMENDATIONS:')
print(f'  1. Increase budget on {top_platform["platform"]} — highest ROAS at {top_platform["roas"]:.2f}x')
print(f'  2. Reduce spend on {worst_platform["platform"]} — lowest ROAS at {worst_platform["roas"]:.2f}x')
print(f'  3. Shift campaign mix toward {top_camptype["campaign_type"]} — best performing type')
print(f'  4. Prioritise {top_industry["industry"]} industry — strongest returns')
print('=' * 60)
