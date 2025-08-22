#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Trains two acceptors for recursive boolean functions, one weighted, one unweighted, and displays them on the 
# console and as state-transition diagrams

func=${1:-xor}
DIR="./examples/data/"

echo -e "\nTraining WFSA acceptor for "$DIR"bool_"$func"_labelled.txt...\n"
python -m hankel.cli learn "$DIR"bool_"$func"_labelled.txt | tee "$DIR"bool_"$func"_wfsa.npz | python -m hankel.cli show -o cons,"$DIR"wfsa.png 

echo -e "\nTraining FSA acceptor for "$DIR"bool_"$func"_labelled.txt...\n"
python -m hankel.cli learn -u -fs "$DIR"bool_"$func"_labelled.txt | tee "$DIR"bool_"$func"_fsa.npz | python -m hankel.cli show -o cons,"$DIR"fsa.png 



