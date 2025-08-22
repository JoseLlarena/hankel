#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Trains two tomita language acceptors, one weighted, one unweighted, and displays them on the console and 
# as state-transition diagrams

number=${1:-2}
DIR="./examples/data/"

echo -e "\nTraining WFSA acceptor for "$DIR"tomita_"$number"_pos_neg_lab.txt...\n"
python -m hankel.cli learn -e 0 "$DIR"tomita_"$number"_pos_neg_lab.txt | tee "$DIR"tomita_"$number"_wfsa.npz | python -m hankel.cli show -o cons,"$DIR"wfsa.png

echo -e "\nTraining FSA acceptor for "$DIR"tomita_"$number"_pos_neg_lab.txt...\n"
python -m hankel.cli learn -u -fs "$DIR"tomita_"$number"_pos_neg_lab.txt | tee "$DIR"tomita_"$number"_fsa.npz | python -m hankel.cli show -o cons,"$DIR"fsa.png