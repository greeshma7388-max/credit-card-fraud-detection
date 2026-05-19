# Credit Card Fraud Detection & Flow-Shop Scheduling

Two optimisation problems: fraud detection using Logistic Regression 
on 284,807 transactions, and job scheduling using a Genetic Algorithm.

## Part A: Credit Card Fraud Detection

### Dataset
284,807 anonymised European credit card transactions (Sep 2013)  
Class imbalance: only 0.17% fraudulent (492 transactions)

### Approach
- Temporal train/test split at 24-hour boundary (prevents data leakage)
- class_weight='balanced' to handle extreme imbalance
- F1-optimised decision threshold (τ* = 0.9923) instead of default 0.5
- Regularisation tuned via 5-fold CV across 6 C values

### Results
| Metric | Value |
|---|---|
| Test ROC-AUC | 0.9779 |
| Test AUPRC | 0.7355 |
| Fraud recall | 80.09% |
| Daily fraud savings | €15,099 (56.2% reduction) |

## Part B: Permutation Flow-Shop Scheduling (Genetic Algorithm)

- Problem: schedule 7 jobs on 4 machines to minimise makespan
- GA with NEH heuristic initialisation, Order Crossover, swap mutation
- Recovered globally optimal sequence [0,4,2,3,6,1,5] — makespan = 60
- 0.00% optimality gap vs Integer Programming reference solution
- Runtime: O(G·P·n·m) vs O(n!) brute force

## Tools & Libraries
Python · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn

## Project Structure
fraud_detection.py     # Logistic Regression pipeline
genetic_algorithm.py   # GA for flow-shop scheduling
