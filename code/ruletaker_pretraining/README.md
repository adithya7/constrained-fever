# Building RuleTaker models

## Pretrained Models

Pretrained RuleTaker models are available at the [link](https://drive.google.com/drive/folders/1MUSYpZkMqQ9JOWJovvXU1L7n31QbOf9z?usp=sharing).

However, if you are interested in retraining the models, follow the below steps.

## RuleTaker-CWA and RuleTaker-Skip-fact datasets

The proposed RuleTaker-CWA and RuleTaker-Skip-fact datasets can be downloaded using this [link](https://drive.google.com/drive/folders/1dznaQruOezoxrsFZKiC1V-vF25FnSEAG?usp=sharing). To learn more about the original RuleTaker dataset, refer to [Clark et al. 2020](https://rule-reasoning.apps.allenai.org).

## Evaluation

```bash
cd code/ruletaker_pretraining
bash run_eval.sh
```

## Training

```bash
cd code/ruletaker_pretraining
bash run_train.sh
```
