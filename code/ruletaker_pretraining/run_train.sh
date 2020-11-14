#!/bin/bash

# ------------------------------------------------- #

python train_rover.py train \
    -input ../build_datasets/ruletaker-cwa/ \
    -checkpoint ./../checkpoints/race_pretrained_models/race_base \
    -bs 8 \
    -save-dir ../../checkpoints/ruletaker_pretrained_models/ruletaker-cwa

# ------------------------------------------------- #

python train_rover.py train \
    -input ../build_datasets/ruletaker-skipfact/ \
    -checkpoint ./../checkpoints/race_pretrained_models/race_base \
    -bs 8 \
    -save-dir ../../checkpoints/ruletaker_pretrained_models/ruletaker-skip-fact

# ------------------------------------------------- #