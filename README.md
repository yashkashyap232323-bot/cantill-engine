# Cantill Data&ML Round 1

## Setup
pip install -r requirements.txt

## Run (one command)
Put company data folders next to this repo (../company_A, ../company_B), then:

    python run.py --all --name yourname

Outputs: outputs/yourname__A.csv, outputs/yourname__B.csv

Single company / new data:

    python run.py --config configs/company_a.json --name yourname --data-dir PATH --suffix A2
    python run.py --config configs/company_c.json --name yourname --suffix C

Add --hide-labels to score against labels (accuracy printed).