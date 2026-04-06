"""
generate_readme.py
==================
Day 14: GitHub Profile README Generator

Industry:  All
Format:    Python script (.py)
Skills:    Python, file I/O, JSON, string templating, markdown

Who uses this:
    Any developer who wants a polished GitHub profile README that
    showcases their project portfolio with consistent formatting.
    Reads projects.json and generates a professional README.md
    ready to push to your GitHub profile repo (username/username).

Usage:
    python generate_readme.py
    Then push generated_profile_README.md to your profile repo.
"""

import json
import os
from datetime import datetime

# ── Load projects ──────────────────────────────────────────────
with open('projects.json', 'r', encoding='utf-8') as f:
    projects = json.load(f)

projects = sorted(projects, key=lambda x: x['day'])

# ── Derived stats ──────────────────────────────────────────────
total_projects  = len(projects)
industries      = sorted(set(p['industry'].split(' / ')[0] for p in projects))
all_tech        = []
for p in projects:
    all_tech.extend(p['tech'])
tech_counts     = {}
for t in all_tech:
    tech_counts[t] = tech_counts.get(t, 0) + 1
top_tech        = sorted(tech_counts.items(), key=lambda x: -x[1])
ipynb_count     = sum(1 for p in projects if p['format'] == 'ipynb')
py_count        = sum(1 for p in projects if p['format'] == 'py')
total_rows      = sum(
    int(p.get('rows', '0').replace(',', ''))
    for p in projects
    if p.get('rows', '0').replace(',', '').isdigit()
)
generated_date  = datetime.today().strftime('%B %d, %Y')

# ── Badge helper ───────────────────────────────────────────────
BADGE_COLORS = {
    'pandas':     '150458', 'numpy':      '013243', 'matplotlib':  '11557c',
    'seaborn':    '4c72b0', 'sqlite3':    '003B57', 'requests':    '2b3137',
    'pytest':     '0a9edc', 'jinja2':     'B41717', 'matplotlib':  '11557c',
}
BADGE_LOGOS = {
    'pandas': 'pandas', 'numpy': 'numpy', 'matplotlib': 'python',
    'seaborn': 'python', 'sqlite3': 'sqlite', 'requests': 'python',
    'pytest': 'pytest',
}

def make_badge(tech):
    tech_lower = tech.lower()
    color = BADGE_COLORS.get(tech_lower, '2b3137')
    logo  = BADGE_LOGOS.get(tech_lower, 'python')
    label = tech.replace('-', '--').replace('_', '__')
    return f'![{tech}](https://img.shields.io/badge/{label}-{color}?style=flat-square&logo={logo}&logoColor=white)'

# ── Format table row ───────────────────────────────────────────
def format_row(p):
    fmt_icon   = '📓' if p['format'] == 'ipynb' else '🐍'
    tech_pills = ' '.join(f'`{t}`' for t in p['tech'])
    return (
        f"| {p['day']:02d} | {fmt_icon} **{p['title']}** | "
        f"{p['industry']} | "
        f"{p['data'].split(' —')[0]} | "
        f"{p.get('rows', '—')} | "
        f"{tech_pills} |"
    )

# ── Build the README ───────────────────────────────────────────
lines = []

# Header
lines.append(f"""# Hi, I'm Harish 👋

> *{total_projects} real-world data projects across {len(industries)} industries — built in {total_projects} consecutive days.*

I'm a data analyst with a background in **Transport, Logistics, and Marketing** who builds end-to-end data projects that solve real business problems — from raw open datasets to actionable insights, using Python, pandas, SQL, and matplotlib.

---

## 📊 28-Day Data Project Streak

**{total_projects} projects completed** · **{ipynb_count} Jupyter notebooks** · **{py_count} Python pipelines** · **{total_rows:,}+ real data rows processed**

### Industries covered
{' · '.join(f'`{i}`' for i in industries)}

### Tech stack used across all projects
{' '.join(make_badge(t) for t, _ in top_tech[:8])}

---

## 🗂️ Project Portfolio

| # | Project | Industry | Dataset | Rows | Stack |
|---|---------|----------|---------|------|-------|""")

# Table rows
for p in projects:
    lines.append(format_row(p))

lines.append("")

# Detailed cards — one per project
lines.append("---\n")
lines.append("## 📁 Project Details\n")

for p in projects:
    fmt_label = 'Jupyter Notebook (.ipynb)' if p['format'] == 'ipynb' else 'Python Script (.py)'
    tech_str  = ' · '.join(f'`{t}`' for t in p['tech'])
    lines.append(f"""### Day {p['day']:02d}: {p['title']}

**Industry:** {p['industry']}  
**Format:** {fmt_label}  
**Stack:** {tech_str}

**Problem:** {p['problem']}

**Key insight:** {p['insight']}

**Data:** {p['data']} · {p.get('rows', 'N/A')} rows

---
""")

# Skills progression
lines.append("""## 📈 Skills Progression

| Skill | Days Used | Projects |
|-------|-----------|----------|""")

skill_projects = {}
for p in projects:
    for t in p['tech']:
        if t not in skill_projects:
            skill_projects[t] = []
        skill_projects[t].append(p['day'])

for tech, days in sorted(skill_projects.items(), key=lambda x: -len(x[1])):
    bar = '█' * len(days) + '░' * (total_projects - len(days))
    lines.append(f"| `{tech}` | {len(days)} | {bar[:20]} |")

lines.append("")

# What I learned
lines.append(f"""---

## 🎓 What I learned across {total_projects} days

- **ETL vs ELT** — built both patterns from scratch, understand when to use each
- **SQL in Python** — sqlite3, SQL views, window functions, aggregate queries
- **Attribution modelling** — first-touch, last-touch, linear, time-decay from scratch
- **Scoring models** — composite weighted normalisation used in Days 3, 6, 7, 13
- **Real data engineering** — chunked loading (2M+ rows), data validation, pipeline logging
- **Pytest** — 39 unit tests written for Day 7 refactor, 39/39 passing
- **Business framing** — every project answers a specific business question for a named stakeholder

---

## 📬 Connect

- 💼 [LinkedIn](https://linkedin.com/in/your-profile)
- 📧 your.email@gmail.com

---

*Generated on {generated_date} by [generate_readme.py](generate_readme.py)*
""")

# ── Write output ───────────────────────────────────────────────
readme_content = '\n'.join(lines)

with open('generated_profile_README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print('=' * 55)
print('GITHUB PROFILE README GENERATOR')
print('=' * 55)
print(f'Projects included:    {total_projects}')
print(f'Industries covered:   {len(industries)}')
print(f'Jupyter notebooks:    {ipynb_count}')
print(f'Python scripts:       {py_count}')
print(f'Total rows processed: {total_rows:,}')
print(f'Top tech stack:       {", ".join(t for t, _ in top_tech[:5])}')
print()
print(f'Output: generated_profile_README.md')
print()
print('Next steps:')
print('  1. Review generated_profile_README.md in VS Code (Ctrl+Shift+V)')
print('  2. Update LinkedIn and email links in the Connect section')
print('  3. Create a repo named exactly: your-github-username/your-github-username')
print('  4. Copy generated_profile_README.md into that repo as README.md')
print('  5. Push — your GitHub profile page will now show this automatically')
print('=' * 55)
