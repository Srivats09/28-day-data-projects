"""
generate_report.py
==================
Day 28: 28-Day Portfolio Impact Report

Industry:  All (Portfolio Summary)
Format:    Python script (.py)
Skills:    Data storytelling, Groq API, pandas, matplotlib, markdown generation

Who uses this:
    Anyone reviewing this GitHub portfolio — recruiters, hiring managers,
    data team leads. This script generates a polished markdown report
    summarising all 28 projects, key metrics, skills, and industries,
    with an AI-written executive summary.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 60)
print('28-DAY PORTFOLIO IMPACT REPORT GENERATOR')
print('=' * 60)

# ══════════════════════════════════════════════════════════════
# PROJECT REGISTRY — All 28 projects
# ══════════════════════════════════════════════════════════════
PROJECTS = [
    {
        'day': 1, 'title': 'MTA Bus Speed & Delay Analyser',
        'industry': 'Transport', 'format': '.ipynb',
        'data': 'MTA NY Open Data', 'real_data': True,
        'skills': ['pandas', 'matplotlib', 'API'],
        'key_metric': 'Slowest routes/boroughs/hours flagged',
        'folder': 'day-01-mta-bus-analyser',
    },
    {
        'day': 2, 'title': 'Retail Basket Association Analyser',
        'industry': 'Retail', 'format': '.ipynb',
        'data': 'UCI Online Retail', 'real_data': True,
        'skills': ['pandas', 'numpy', 'association rules'],
        'key_metric': 'Lift 15.03x · £29,030 revenue opportunity',
        'folder': 'day-02-retail-basket-analyser',
    },
    {
        'day': 3, 'title': 'Hospital Readmission Risk Scorer',
        'industry': 'Healthcare', 'format': '.ipynb',
        'data': 'CMS HRRP', 'real_data': True,
        'skills': ['pandas', 'seaborn', 'risk scoring'],
        'key_metric': '202 hospitals scored · Oroville highest risk 8.13/10',
        'folder': 'day-03-hospital-readmission-risk',
    },
    {
        'day': 4, 'title': 'Smart Meter Energy ETL',
        'industry': 'Energy', 'format': '.py',
        'data': 'UCI Household Power', 'real_data': True,
        'skills': ['pandas', 'SQLite', 'ETL', 'chunked processing'],
        'key_metric': '2.07M rows · peak demand 20:00 · 4/4 validations passed',
        'folder': 'day-04-smart-meter-etl',
    },
    {
        'day': 5, 'title': 'Ad Campaign SQL Dashboard',
        'industry': 'Marketing', 'format': '.ipynb',
        'data': 'Synthetic', 'real_data': False,
        'skills': ['pandas', 'SQLite', 'SQL', 'matplotlib'],
        'key_metric': 'Overall ROAS 6.04x · Email ROAS 178.56x',
        'folder': 'day-05-ad-campaign-sql-dashboard',
    },
    {
        'day': 6, 'title': 'Employee Attrition Analyser',
        'industry': 'HR', 'format': '.ipynb',
        'data': 'IBM HR Dataset', 'real_data': True,
        'skills': ['pandas', 'seaborn', 'risk scoring'],
        'key_metric': '16.1% attrition · $20.4M turnover cost · 278 at-risk flagged',
        'folder': 'day-06-employee-attrition-analyser',
    },
    {
        'day': 7, 'title': 'Refactored Attrition Module + Tests',
        'industry': 'HR', 'format': '.py',
        'data': 'IBM HR Dataset', 'real_data': True,
        'skills': ['pytest', 'refactoring', 'modular code'],
        'key_metric': '39/39 tests passing · 7 clean functions',
        'folder': 'day-07-refactor-attrition',
    },
    {
        'day': 8, 'title': 'UK Property Price ETL',
        'industry': 'Real Estate', 'format': '.py',
        'data': 'UK Land Registry', 'real_data': True,
        'skills': ['pandas', 'SQLite', 'ETL'],
        'key_metric': '920k transactions · new build 20.4% premium',
        'folder': 'day-08-property-price-etl',
    },
    {
        'day': 9, 'title': 'Government SLA Breach Detector',
        'industry': 'Government', 'format': '.ipynb',
        'data': 'Chicago 311', 'real_data': True,
        'skills': ['pandas', 'API', 'SLA analysis'],
        'key_metric': '48.9% breach rate · 1.55M excess citizen wait-days',
        'folder': 'day-09-logistics-sla-breach-detector',
    },
    {
        'day': 10, 'title': 'Streaming Content ELT Pipeline',
        'industry': 'Media', 'format': '.py',
        'data': 'Netflix Titles', 'real_data': True,
        'skills': ['pandas', 'SQLite', 'ELT', 'SQL views'],
        'key_metric': '6,234 titles · USA top country · 90-120 min sweet spot',
        'folder': 'day-10-streaming-content-elt',
    },
    {
        'day': 11, 'title': 'Marketing Attribution Analyser',
        'industry': 'Marketing', 'format': '.ipynb',
        'data': 'Synthetic', 'real_data': False,
        'skills': ['pandas', 'attribution modelling', 'matplotlib'],
        'key_metric': '4 attribution models · Organic SEO 38.4% conv rate',
        'folder': 'day-11-marketing-attribution',
    },
    {
        'day': 12, 'title': 'Pharmacy Stock Alert System',
        'industry': 'Healthcare', 'format': '.ipynb',
        'data': 'FDA Drug Shortage API', 'real_data': True,
        'skills': ['pandas', 'API', 'alert scoring'],
        'key_metric': '286 critical shortages · Fentanyl 5,207 days',
        'folder': 'day-12-pharmacy-stock-alert',
    },
    {
        'day': 13, 'title': 'Supplier Risk Ranker',
        'industry': 'Logistics', 'format': '.ipynb',
        'data': 'USAID SCMS', 'real_data': True,
        'skills': ['pandas', 'risk scoring', 'seaborn'],
        'key_metric': '11.5% late rate · Ocean 17.5% vs Air 9.6%',
        'folder': 'day-13-supplier-risk-ranker',
    },
    {
        'day': 14, 'title': 'GitHub README Generator',
        'industry': 'All', 'format': '.py',
        'data': 'projects.json', 'real_data': False,
        'skills': ['Python', 'JSON', 'markdown generation'],
        'key_metric': 'Auto-generates README from project metadata',
        'folder': 'day-14-github-readme-generator',
    },
    {
        'day': 15, 'title': 'Student Performance Visualiser',
        'industry': 'Education', 'format': '.ipynb',
        'data': 'UCI Student Data', 'real_data': True,
        'skills': ['pandas', 'seaborn', 'correlation analysis'],
        'key_metric': '22% fail rate · G2 r=0.911 · 270 at-risk flagged',
        'folder': 'day-15-student-performance-visualiser',
    },
    {
        'day': 16, 'title': 'Uber & Lyft Surge Pricing Analyser',
        'industry': 'Retail', 'format': '.ipynb',
        'data': 'Kaggle Rideshare', 'real_data': True,
        'skills': ['pandas', 'matplotlib', 'pricing analysis'],
        'key_metric': '637k rides · Lyft $1.55 premium · 3.3% surge rate',
        'folder': 'day-16-surge-pricing-analyser',
    },
    {
        'day': 17, 'title': 'Logistics Carbon Footprint ETL',
        'industry': 'Logistics', 'format': '.py',
        'data': 'BEIS 2023 (embedded)', 'real_data': True,
        'skills': ['pandas', 'SQLite', 'ETL', 'carbon accounting'],
        'key_metric': '6,654t CO2e · Air 98.2% of emissions · 97.4% saving potential',
        'folder': 'day-17-logistics-carbon-etl',
    },
    {
        'day': 18, 'title': 'Customer Sentiment ETL',
        'industry': 'Marketing', 'format': '.py',
        'data': 'TripAdvisor + Groq API', 'real_data': True,
        'skills': ['pandas', 'Groq API', 'LLaMA', 'SQLite', 'NLP'],
        'key_metric': '214 reviews tagged · 71.5% positive · AI-powered tagging',
        'folder': 'day-18-customer-sentiment-etl',
    },
    {
        'day': 19, 'title': 'HR Pay Equity Analyser',
        'industry': 'HR', 'format': '.ipynb',
        'data': 'UK Gender Pay Gap Gov', 'real_data': True,
        'skills': ['pandas', 'seaborn', 'equity analysis'],
        'key_metric': '10,767 UK employers · 11.5% mean gap · glass ceiling confirmed',
        'folder': 'day-19-pay-equity-analyser',
    },
    {
        'day': 20, 'title': 'Real Estate Rental Yield Calculator',
        'industry': 'Real Estate', 'format': '.ipynb',
        'data': 'ONS IPHRP Excel', 'real_data': True,
        'skills': ['pandas', 'openpyxl', 'financial modelling'],
        'key_metric': '11 UK regions · 4.5% mortgage kills 10/11 net yields',
        'folder': 'day-20-rental-yield-calculator',
    },
    {
        'day': 21, 'title': 'Energy Demand Forecasting',
        'industry': 'Energy', 'format': '.py',
        'data': 'National Grid ESO', 'real_data': True,
        'skills': ['pandas', 'numpy', 'time series', 'forecasting'],
        'key_metric': '4,176 real readings · 7.24% MAPE · 7-day forecast',
        'folder': 'day-21-energy-demand-forecasting',
    },
    {
        'day': 22, 'title': 'Media Genre SQL Ranker',
        'industry': 'Media', 'format': '.ipynb',
        'data': 'MovieLens', 'real_data': True,
        'skills': ['pandas', 'SQLite', 'SQL window functions'],
        'key_metric': '9,742 movies · IMAX 98.3 opportunity score',
        'folder': 'day-22-media-genre-sql-ranker',
    },
    {
        'day': 23, 'title': 'Edtech Dropout Early Warning',
        'industry': 'Education', 'format': '.ipynb',
        'data': 'OULAD Open University', 'real_data': True,
        'skills': ['pandas', 'SQLite', 'risk scoring', 'seaborn'],
        'key_metric': '32,593 students · 99.9% critical tier accuracy · 11,722 flagged',
        'folder': 'day-23-edtech-dropout-warning',
    },
    {
        'day': 24, 'title': 'Retail Demand Forecasting ETL',
        'industry': 'Retail', 'format': '.py',
        'data': 'Supermarket Sales', 'real_data': True,
        'skills': ['pandas', 'numpy', 'SQLite', 'forecasting'],
        'key_metric': '1,000 transactions · 4-week forecast · 6 product lines',
        'folder': 'day-24-retail-demand-forecasting',
    },
    {
        'day': 25, 'title': 'Healthcare Claims Anomaly Detector',
        'industry': 'Healthcare', 'format': '.ipynb',
        'data': 'CMS Medicare API', 'real_data': True,
        'skills': ['pandas', 'numpy', 'Z-score detection', 'seaborn'],
        'key_metric': '149,767 providers · $9.9B flagged · 5.7% anomaly rate',
        'folder': 'day-25-healthcare-claims-anomaly',
    },
    {
        'day': 26, 'title': 'Social Media ROI ETL Pipeline',
        'industry': 'Marketing', 'format': '.py',
        'data': 'Synthetic', 'real_data': False,
        'skills': ['pandas', 'SQLite', 'marketing analytics', 'matplotlib'],
        'key_metric': '458 campaigns · Meta 5.24x ROAS · $12.7M profit',
        'folder': 'day-26-social-media-roi-etl',
    },
    {
        'day': 27, 'title': 'IoT Sensor Anomaly Detector',
        'industry': 'Energy', 'format': '.ipynb',
        'data': 'NASA CMAPSS', 'real_data': True,
        'skills': ['pandas', 'numpy', 'rolling Z-score', 'predictive maintenance'],
        'key_metric': '100 engines · s11 r=0.696 · 20.7% critical anomaly rate',
        'folder': 'day-27-iot-sensor-anomaly',
    },
    {
        'day': 28, 'title': '28-Day Portfolio Impact Report',
        'industry': 'All', 'format': '.py',
        'data': 'Project registry', 'real_data': False,
        'skills': ['Python', 'Groq API', 'matplotlib', 'markdown', 'data storytelling'],
        'key_metric': '28 projects · 12 industries · 25/28 real datasets',
        'folder': 'day-28-portfolio-impact-report',
    },
]

df = pd.DataFrame(PROJECTS)
print(f'\n[DATA] Project registry loaded: {len(df)} projects')


# ══════════════════════════════════════════════════════════════
# ANALYSE — Portfolio statistics
# ══════════════════════════════════════════════════════════════
print('\n[ANALYSE] Computing portfolio statistics...')

total_projects   = len(df)
real_data_count  = df['real_data'].sum()
industries       = df['industry'].nunique()
py_count         = (df['format'] == '.py').sum()
ipynb_count      = (df['format'] == '.ipynb').sum()

all_skills = []
for skills in df['skills']:
    all_skills.extend(skills)
from collections import Counter
skill_counts = Counter(all_skills)
top_skills   = skill_counts.most_common(10)

industry_counts = df['industry'].value_counts()

print(f'  Total projects:      {total_projects}')
print(f'  Real datasets:       {real_data_count}/{total_projects}')
print(f'  Industries:          {industries}')
print(f'  Python scripts:      {py_count}')
print(f'  Jupyter notebooks:   {ipynb_count}')
print(f'\n  Top skills: {top_skills[:5]}')
print(f'\n  Industry breakdown:')
print(industry_counts.to_string())


# ══════════════════════════════════════════════════════════════
# VISUALISE — Portfolio summary chart
# ══════════════════════════════════════════════════════════════
print('\n[VISUALISE] Building portfolio chart...')

fig = plt.figure(figsize=(18, 12))
fig.suptitle('28-Day Data Projects Portfolio — Impact Summary',
             fontsize=16, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

INDUSTRY_COLORS = {
    'Transport'  : '#378ADD', 'Retail'     : '#1D9E75',
    'Healthcare' : '#E24B4A', 'Energy'     : '#EF9F27',
    'Marketing'  : '#534AB7', 'HR'         : '#B5D4F4',
    'Real Estate': '#085041', 'Logistics'  : '#888780',
    'Government' : '#F4A261', 'Media'      : '#E76F51',
    'Education'  : '#2A9D8F', 'All'        : '#264653',
}

# Panel 1 — Projects by industry
ax1 = fig.add_subplot(gs[0, 0])
ind_sorted = industry_counts.sort_values(ascending=True)
colors1    = [INDUSTRY_COLORS.get(i, '#888') for i in ind_sorted.index]
ax1.barh(ind_sorted.index, ind_sorted.values, color=colors1)
ax1.set_xlabel('Number of projects')
ax1.set_title('Projects by industry')
ax1.tick_params(axis='y', labelsize=8)

# Panel 2 — Format breakdown pie
ax2 = fig.add_subplot(gs[0, 1])
format_counts = df['format'].value_counts()
ax2.pie(format_counts.values, labels=['.py scripts', '.ipynb notebooks'],
        autopct='%1.0f%%', colors=['#378ADD', '#1D9E75'], startangle=90)
ax2.set_title('Format breakdown')

# Panel 3 — Real vs synthetic data
ax3 = fig.add_subplot(gs[0, 2])
data_counts = df['real_data'].value_counts()
ax3.pie([real_data_count, total_projects - real_data_count],
        labels=['Real data', 'Synthetic'],
        autopct='%1.0f%%', colors=['#1D9E75', '#EF9F27'], startangle=90)
ax3.set_title('Data source breakdown')

# Panel 4 — Top skills bar chart
ax4 = fig.add_subplot(gs[1, 0])
skill_names  = [s[0] for s in top_skills]
skill_values = [s[1] for s in top_skills]
skill_colors = ['#378ADD' if i < 3 else '#1D9E75' if i < 6 else '#EF9F27'
                for i in range(len(skill_names))]
ax4.barh(skill_names[::-1], skill_values[::-1], color=skill_colors[::-1])
ax4.set_xlabel('Times used')
ax4.set_title('Top 10 skills across 28 projects')
ax4.tick_params(axis='y', labelsize=9)

# Panel 5 — Project timeline (day vs industry)
ax5 = fig.add_subplot(gs[1, 1:])
y_map     = {ind: i for i, ind in enumerate(sorted(df['industry'].unique()))}
for _, row in df.iterrows():
    color  = INDUSTRY_COLORS.get(row['industry'], '#888')
    marker = 's' if row['format'] == '.py' else 'o'
    ax5.scatter(row['day'], y_map[row['industry']], color=color,
                s=120, marker=marker, zorder=3)
    ax5.text(row['day'], y_map[row['industry']] + 0.3,
             str(row['day']), ha='center', fontsize=6.5, color='#333')
ax5.set_yticks(list(y_map.values()))
ax5.set_yticklabels(list(y_map.keys()), fontsize=8)
ax5.set_xlabel('Day')
ax5.set_title('Project timeline by industry\n(■ = .py script, ● = .ipynb notebook)')
ax5.set_xlim(0, 29)
ax5.grid(True, alpha=0.3)

plt.savefig(f'{OUTPUT_DIR}/portfolio_chart.png', dpi=150, bbox_inches='tight')
print('  Chart saved as portfolio_chart.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# AI EXECUTIVE SUMMARY — Groq API
# ══════════════════════════════════════════════════════════════
print('\n[AI] Generating executive summary with Groq/LLaMA...')

api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    raise ValueError('GROQ_API_KEY not set. Run: $env:GROQ_API_KEY = "your-key"')

client = Groq(api_key=api_key)

project_summary = '\n'.join([
    f"Day {p['day']}: {p['title']} ({p['industry']}) — {p['key_metric']}"
    for p in PROJECTS
])

prompt = f"""You are writing a professional portfolio executive summary for a data analyst's GitHub profile.
They completed 28 data projects in 28 consecutive days across 12 industries.
Here are all 28 projects with their key results:

{project_summary}

Write a compelling 3-paragraph executive summary (250-300 words) that:
1. Opens with the overall achievement and scale (28 days, 12 industries, real datasets)
2. Highlights 4-5 of the most impressive specific metrics from the projects
3. Closes with the demonstrated skillset and what this portfolio shows about the analyst

Write in third person. Be specific with numbers. Sound professional but not robotic.
Do not use bullet points — write in flowing paragraphs."""

message = client.chat.completions.create(
    model='llama-3.1-8b-instant',
    max_tokens=600,
    messages=[
        {'role': 'system', 'content': 'You are a professional technical writer creating portfolio summaries for data analysts.'},
        {'role': 'user', 'content': prompt}
    ]
)
executive_summary = message.choices[0].message.content.strip()
print(f'\n  Executive summary generated ({len(executive_summary.split())} words)')


# ══════════════════════════════════════════════════════════════
# GENERATE MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════
print('\n[REPORT] Generating portfolio markdown report...')

skills_section = '\n'.join([f'- **{s}** — used in {c} projects' for s, c in top_skills])

industry_section = '\n'.join([
    f'| {ind} | {cnt} |'
    for ind, cnt in industry_counts.items()
])

projects_table = '\n'.join([
    f"| {p['day']:02d} | [{p['title']}](https://github.com/Srivats09/28-day-data-projects/tree/main/{p['folder']}) "
    f"| {p['industry']} | {p['format']} | {p['data']} | {p['key_metric']} |"
    for p in PROJECTS
])

report = f"""# 28-Day Data Projects Portfolio

> **28 projects · 28 consecutive days · 12 industries · 25 real datasets**

![Portfolio Chart](output/portfolio_chart.png)

---

## Executive Summary

{executive_summary}

---

## Portfolio Statistics

| Metric | Value |
|--------|-------|
| Total projects | {total_projects} |
| Real-world datasets | {real_data_count}/{total_projects} ({real_data_count/total_projects*100:.0f}%) |
| Industries covered | {industries} |
| Python scripts (.py) | {py_count} |
| Jupyter notebooks (.ipynb) | {ipynb_count} |
| Consecutive days | 28 |

---

## Skills Used

{skills_section}

---

## Industries Covered

| Industry | Projects |
|----------|---------|
{industry_section}

---

## All 28 Projects

| Day | Project | Industry | Format | Data | Key Result |
|-----|---------|----------|--------|------|------------|
{projects_table}

---

## Tech Stack

**Languages:** Python 3.14  
**Analysis:** pandas · numpy · scipy  
**Visualisation:** matplotlib · seaborn  
**Databases:** SQLite · SQL window functions  
**APIs:** CMS Medicare · FDA · National Grid ESO · Groq/LLaMA · Chicago 311 · FDA · MTA  
**ML/Stats:** Z-score detection · association rules · linear trend forecasting · risk scoring  
**ETL:** Extract · Transform · Load pipelines with validation  
**Testing:** pytest (39/39 tests, Day 7)

---

## How to Run Any Project

```bash
git clone https://github.com/Srivats09/28-day-data-projects.git
cd 28-day-data-projects/day-XX-project-name
pip install -r requirements.txt
python download.py        # fetch data (where applicable)
jupyter notebook analysis.ipynb   # or: python etl_pipeline.py
```

---

*Generated automatically by Day 28 portfolio report script — {pd.Timestamp.now().strftime('%Y-%m-%d')}*
"""

report_path = f'{OUTPUT_DIR}/PORTFOLIO_REPORT.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'  Report saved as {report_path}')


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('PORTFOLIO IMPACT SUMMARY')
print('=' * 60)
print(f'Total projects completed:  {total_projects}')
print(f'Consecutive days:          28')
print(f'Industries covered:        {industries}')
print(f'Real datasets used:        {real_data_count}/{total_projects} ({real_data_count/total_projects*100:.0f}%)')
print(f'Python scripts:            {py_count}')
print(f'Jupyter notebooks:         {ipynb_count}')
print(f'\nTop 5 skills demonstrated:')
for skill, count in top_skills[:5]:
    print(f'  {skill:25s}: {count} projects')
print(f'\nIndustry coverage:')
for ind, cnt in industry_counts.items():
    print(f'  {ind:20s}: {cnt} projects')
print(f'\nStandout results across 28 days:')
print(f'  149,767 Medicare providers anomaly-scanned ($9.9B flagged)')
print(f'  32,593 students dropout-risk scored (99.9% critical tier)')
print(f'  10,767 UK employers pay-gap analysed')
print(f'  20,631 NASA turbofan sensor readings processed')
print(f'  100,836 MovieLens ratings ranked by SQL window functions')
print(f'  2.07M smart meter rows ETL-processed in chunks')
print(f'\nPortfolio report: {report_path}')
print(f'Portfolio chart:  {OUTPUT_DIR}/portfolio_chart.png')
print('=' * 60)
print('\n🎉 28-DAY STREAK COMPLETE!')
