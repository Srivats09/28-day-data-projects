# Day 7: Refactored Attrition Analysis Module

**Industry:** HR / Recruitment  
**Format:** Python module (.py) + pytest test suite  
**Skills:** modular design · docstrings · pytest · pandas · seaborn · scoring model

## What this is
A refactored, production-grade version of the Day 6 attrition 
analysis — rebuilt as a testable Python module with clean function 
signatures, full docstrings, and 39 passing unit tests.

## Why refactor matters
A notebook runs top to bottom once. A module can be imported, 
scheduled, extended, and tested independently. This is how data 
pipelines are written in production environments.

## Structure
```
attrition_analysis.py   ← 7 clean functions + main()
test_attrition.py       ← 39 pytest unit tests, 5 test classes
```

## Functions
- `load_data()` — load CSV with file validation
- `clean_data()` — encode, band, enrich, copy-safe
- `calculate_attrition_rate()` — groupby attrition rate with error handling
- `calculate_turnover_cost()` — 1.5x salary cost model, configurable multiplier
- `score_intervention_risk()` — weighted rule-based risk scoring
- `compare_satisfaction()` — stayed vs left satisfaction comparison
- `plot_dashboard()` — 6-panel seaborn + matplotlib chart
- `export_results()` — structured CSV output

## Test results
```
39 passed in 2.87s
```

## How to run
```bash
pip install -r requirements.txt
python attrition_analysis.py
pytest test_attrition.py -v
```
```

