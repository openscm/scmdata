#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
NOTEBOOK_DIR="$SCRIPT_DIR/../notebooks"

for nb in $NOTEBOOK_DIR/*.ipynb; do
  echo "running $nb"
  jupyter nbconvert \
    --inplace \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=60000 \
    "$nb"

done