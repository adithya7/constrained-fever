# BERT-concat

## Pretrained Models

Pretrained FEVER models are available [here](https://drive.google.com/drive/folders/1cufeowvAbJ-ta7Uqm3BFnStYpAYsF-C6?usp=sharing).

However, if you are interested in evaluating or retraining, follow the below steps.

## Evaluation

To run the evaluation, first download the [evaluation sets](https://drive.google.com/drive/folders/1zthG6CS2HteYyj59gmrIthblG0ovSkzd?usp=sharing) to `../../eval/` and then run `run_eval.sh`.

```bash
# download evaluation sets to ../../eval/
cd code/bert-concat
bash run_eval.sh
```

**Note:** Due to randomized initialization of _abstract entity markers\`_ embeddings, the results of `Anon.` evaluation set might slightly vary when compared to the results reported in our paper.

## Training

We use a simple BERT-based classifier on the claim and extracted evidence sentences. For the evidence extraction, we use the state-of-the-art retrieval results from [KGAT](https://github.com/thunlp/KernelGAT). First download the train and dev files from [here](https://drive.google.com/file/d/1jyrTkU9HB4nyay6jG1_xj-8t8NAQbrWg/view?usp=sharing).

```bash
# Original training strategy
# as downloaded above
export INPUT_PATH='fever_train'
# add path to save checkpoints
export MODEL_PATH=''
python train.py train \
    -input ${INPUT_PATH} \
    -bs 16 \
    -save-dir ${MODEL_PATH} \
    -eval-steps 1000

# CWA/Skip-fact training strategy
# provide the checkpoint to the relevant RuleTaker fine-tuned BERT
export CHECKPOINT_PATH=''
python train.py train \
    -input ${INPUT_PATH} \
    -checkpoint ${CHECKPOINT_PATH} \
    -bs 16 \
    -save-dir ${MODEL_PATH} \
    -eval-steps 1000
```