#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Trains two acceptors for the Reber grammar, one weighted, one unweighted, and displays them on the 
# console and as state-transition diagrams

DIR="./examples/data/"

echo -e "\nTraining WFSA acceptor for "$DIR"reber_pos_neg_lab.txt...\n"
python -m hankel.cli learn -b length -tp 14:15:1 -ts 14:15:1 "$DIR"reber_pos_neg_lab.txt | tee "$DIR"reber_wfsa.npz | python -m hankel.cli show -o cons,"$DIR"wfsa.png 

echo -e "\nTraining FSA acceptor for "$DIR"reber_pos_neg_lab.txt...\n"
python -m hankel.cli learn -b length -tp 14:15:1 -ts 14:15:1 -u "$DIR"reber_pos_neg_lab.txt | tee "$DIR"reber_fsa.npz | python -m hankel.cli show -o cons,"$DIR"fsa.png 



