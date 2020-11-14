# RACE pretraining

## Pretrained Model

Pretrained RACE QA models are available [here](https://drive.google.com/drive/folders/1yZGtw0rxe5NzpypkKCwF0Guq3LteVkjK?usp=sharing).

However, if you are interested in retraining, follow the below steps.

## Download RACE dataset

Firstly, download the RACE question answering dataset from [here](https://www.cs.cmu.edu/~glai1/data/race/).

## Training

We adapt the BERT for multi-choice QA from [huggingface transformers repo](https://github.com/huggingface/transformers/blob/master/examples/multiple-choice/run_multiple_choice.py)

```bash
export RACE_DATA_PATH=''
export MODEL_PATH=''
CUDA_VISIBLE_DEVICES=0 python -W ignore run_multiple_choice.py \
    --task_name race \
    --model_name_or_path bert-base-cased \
    --do_train \
    --do_eval \
    --data_dir ${RACE_DATA_PATH} \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --output_dir ${MODEL_PATH} \
    --per_gpu_eval_batch_size=2 \
    --per_gpu_train_batch_size=2 \
    --gradient_accumulation_steps 8 \
    --overwrite_output
```
