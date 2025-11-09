#!/bin/sh
# Shell script
echo Starting heuristic parameter identification approach based on FMECA data
cd C:\pyhthon_envs\python_bayesian_network\Scripts && .\activate
cd C:\Users\yanni\OneDrive\Promotion\03_Content\05_Publikationen\BN_KRR\GitHub\BDSN-FDC
python src/heuristic_parameter_identification/heuristic_parameter_identification.py data/fmeca_data.csv --verbose=1