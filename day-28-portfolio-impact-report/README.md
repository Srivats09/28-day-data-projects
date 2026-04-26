# Day 28: 28-Day Portfolio Impact Report

**Industry:** All  
**Format:** Python script (.py)  
**Skills:** pandas · matplotlib · Groq API · LLaMA · markdown generation · data storytelling

## What this is
The final project in the 28-day streak — a script that reads the
full project registry, computes portfolio statistics, generates a
4-panel summary chart, and uses Groq/LLaMA to write an AI-generated
executive summary. Output is a polished markdown portfolio report.

## Portfolio Stats
- Total projects: 28 across 28 consecutive days
- Industries covered: 12
- Real datasets used: 23/28 (82%)
- Python scripts: 11 | Jupyter notebooks: 17

## Top Skills Demonstrated
- pandas (25 projects) · SQLite (10) · seaborn (7) · matplotlib (6) · numpy (5)

## Standout Results
- 149,767 Medicare providers anomaly-scanned ($9.9B flagged)
- 32,593 students dropout-risk scored (99.9% critical tier accuracy)
- 10,767 UK employers gender pay-gap analysed
- 20,631 NASA turbofan sensor readings processed
- 2.07M smart meter rows ETL-processed in chunks

## How to run
```bash
pip install -r requirements.txt
$env:GROQ_API_KEY = "your-key"
python generate_report.py
```