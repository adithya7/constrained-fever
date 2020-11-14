#!/bin/bash

# for the reasoning architecture `bert-concat`
#   for each of [`baseline`, `cwa`, `skip-fact`] encoder versions
#     for each of [`Std.`, `Symm.`, `Anon.`] evaluation sets

# ------------------------------------------------- #

mkdir -p eval_predictions

CUDA_VISIBLE_DEVICES=0 python -W ignore train.py eval \
  -input ../../eval/dev_retrievedsents_leetal.json \
  -model ../../checkpoints/fever_models/bert/baseline \
  -task fever \
  -bs 16 \
  -out eval_predictions/baseline_std_dev.jsonl

CUDA_VISIBLE_DEVICES=0 python -W ignore train.py eval \
  -input ../../eval/fever_symmetric_dev_kgat.jsonl \
  -model ../../checkpoints/fever_models/bert/baseline \
  -task fever \
  -bs 16 \
  -out eval_predictions/baseline_symm_dev.jsonl

CUDA_VISIBLE_DEVICES=0 python -W ignore train.py eval \
  -input ../../eval/anon_bert_dev.json \
  -model ../../checkpoints/fever_models/bert/baseline \
  -task fever \
  -bs 16 \
  -out eval_predictions/baseline_anon_dev.jsonl

## ------------------------------------------------- #

CUDA_VISIBLE_DEVICES=0 python -W ignore train.py eval \
  -input ../../eval/dev_retrievedsents_leetal.json \
  -model ../../checkpoints/fever_models/bert/cwa \
  -task fever \
  -bs 16 \
  -out eval_predictions/cwa_std_dev.jsonl

CUDA_VISIBLE_DEVICES=0 python -W ignore train.py eval \
  -input ../../eval/fever_symmetric_dev_kgat.jsonl \
  -model ../../checkpoints/fever_models/bert/cwa \
  -task fever \
  -bs 16 \
  -out eval_predictions/cwa_symm_dev.jsonl

CUDA_VISIBLE_DEVICES=0 python -W ignore train.py eval \
  -input ../../eval/anon_bert_dev.json \
  -model ../../checkpoints/fever_models/bert/cwa \
  -task fever \
  -bs 16 \
  -out eval_predictions/cwa_anon_dev.jsonl

## ------------------------------------------------- #

CUDA_VISIBLE_DEVICES=0 python -W ignore train.py eval \
  -input ../../eval/dev_retrievedsents_leetal.json \
  -model ../../checkpoints/fever_models/bert/skip-fact \
  -task fever \
  -bs 16 \
  -out eval_predictions/skip-fact_std_dev.jsonl

CUDA_VISIBLE_DEVICES=0 python -W ignore train.py eval \
  -input ../../eval/fever_symmetric_dev_kgat.jsonl \
  -model ../../checkpoints/fever_models/bert/skip-fact \
  -task fever \
  -bs 16 \
  -out eval_predictions/skip-fact_symm_dev.jsonl

CUDA_VISIBLE_DEVICES=0 python -W ignore train.py eval \
  -input ../../eval/anon_bert_dev.json \
  -model ../../checkpoints/fever_models/bert/skip-fact \
  -task fever \
  -bs 16 \
  -out eval_predictions/skip-fact_anon_dev.jsonl

# ------------------------------------------------- #