#!/bin/bash


# ------------------------------------------------- #

CUDA_VISIBLE_DEVICES=0 python -W ignore train_rover.py eval \
  -input ../build_datasets/ruletaker-cwa/test.jsonl \
  -model ../../checkpoints/ruletaker_pretrained_models/ruletaker-cwa \
  -task rover \
  -bs 64

# ------------------------------------------------- #

CUDA_VISIBLE_DEVICES=0 python -W ignore train_rover.py eval \
  -input ../build_datasets/ruletaker-skipfact/test.jsonl \
  -model ../../checkpoints/ruletaker_pretrained_models/ruletaker-skip-fact \
  -task rover \
  -bs 64

# ------------------------------------------------- #