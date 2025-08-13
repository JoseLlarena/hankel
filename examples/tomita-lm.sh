#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Trains a WFSA LM for a tomita language, shows it on the console and as state-diagram, and predicts the most and least 
# likely sequences from another tomita language

train=${1:-2}
predict=${2:-4}
DIR="./examples/data/"

if [[ $# -eq 1 ]]; then
    echo "Error: Please provide both training and prediction tomita numbers, or neither (defaults to 4 and 6)" >&2
    exit 1
fi  

echo -e "\nTraining LM for tomita_"$train"_unlabelled.txt...\n"
python -m hankel.cli learn -k lm ppl -e 0 "$DIR"tomita_"$train"_unlabelled.txt | tee "$DIR"tomita_"$train"_lm.npz | python -m hankel.cli show -o cons,"$DIR"lm.png

echo -e "Predicting "$DIR"tomita_"$predict"_unlabelled.txt with "$DIR"tomita_"$train"_lm.npz...\n"
python -m hankel.cli predict -s -o "$DIR"pred_tomita_"$predict"_by_"$train".txt "$DIR"tomita_"$predict"_unlabelled.txt "$DIR"tomita_"$train"_lm.npz

echo -e "25 LEAST LIKELY\t(base-10 nll then length then alphabetically)\n"
head -n25 "$DIR"pred_tomita_"$predict"_by_"$train".txt
printf '%*s\n' "$(tput cols)" '' | tr ' ' '-'

echo -e "25 MOST LIKELY\t(base-10 nll then length then alphabetically)\n"
tac  "$DIR"pred_tomita_"$predict"_by_"$train".txt | head -n25