# Day 2: Retail Basket Association Analyser

**Industry:** Retail / E-commerce  
**Format:** Jupyter Notebook  
**Skills:** pandas · numpy · itertools · matplotlib · association rules

## Who uses this
A category manager deciding which products to bundle, co-locate,
or cross-promote — driven by transaction data, not gut feel.

## Problem
Retailers leave cross-sell revenue on the table without basket
analysis. This notebook finds which products are genuinely bought
together using support, confidence, and lift metrics built from scratch.

## Dataset
UCI Online Retail Dataset — 541,909 real UK e-commerce transactions  
Source: archive.ics.uci.edu (no login required)

## Key Findings
- Strongest pair: GREEN REGENCY TEACUP AND SAUCER + PINK REGENCY TEACUP AND SAUCER (lift 15.03x)
- Highest revenue opportunity: GREEN + PINK REGENCY TEACUP AND SAUCER — £29,030.81
- Rules with lift > 2: 655 out of 4,942 total rules
- Recommendation: Bundle all 3 Regency teacup variants as a set

## Output
![Basket Analysis](basket_analysis.png)

## How to run
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

Then commit and push:
```
Day 2: retail basket association analyser — pandas + numpy + lift 15x on teacup pairs