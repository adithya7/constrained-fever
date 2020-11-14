# Constructing RuleTaker variants for fact verification

Scripts to build RuleTaker-CWA and RuleTaker-Skip-fact from the original RuleTaker dataset [Clark et al. 2020](https://arxiv.org/abs/2002.05867).

First download the RuleTaker dataset,

```bash
wget http://data.allenai.org/rule-reasoning/rule-reasoning-dataset-V2020.2.4.zip
unzip rule-reasoning-dataset-V2020.2.4.zip
```

## Prepare RuleTaker-CWA

To prepare RuleTaker-CWA dataset,

```bash
mkdir ruletaker-cwa
for split in train dev test; do
    python prepare_RuleTaker_CWA.py \
        rule-reasoning-dataset-V2020.2.4/depth-3ext-NatLang/${split}.jsonl \
        rule-reasoning-dataset-V2020.2.4/depth-3ext-NatLang/meta-${split}.jsonl \
        ruletaker-cwa/${split}.jsonl
done
```

## Prepare RuleTaker-Skip-fact

**Note**: due to the inherent randomness in the algorithm, you might get slightly different skip-fact variants in each run. To reproduce the numbers in the original paper, please use the released RuleTaker-Skip-fact dataset ([link](https://drive.google.com/file/d/10QqJLTN5MKcm_xvcEgQnXT4EvOme6jdS/view?usp=sharing)).

To prepare a RuleTaker-Skip-fact dataset,

```bash
mkdir ruletaker-skipfact
for split in train dev test; do
    python prepare_RuleTaker_Skipfact.py \
        rule-reasoning-dataset-V2020.2.4/depth-3ext-NatLang/${split}.jsonl \
        rule-reasoning-dataset-V2020.2.4/depth-3ext-NatLang/meta-${split}.jsonl \
        ruletaker-skipfact/${split}.jsonl
done
```

## Creating entity anonymized FEVER dataset

Script for doing so is [create_data_anonymized.py](create_data_anonymized.py). Run the script as follows:

```bash
python create_data_anonymized.py ./bert_dev.json ./anon_dev.json
python create_data_anonymized.py ./bert_train.json ./anon_train.json
```

wherein the first argument is the input file and the second one is the output file. Typically, the input file is the one that is produced after sentence-retrieval step in FEVER task. The `./bert_dev.json` and `./bert_train.json` files can be downloaded from the original [KGAT drive folder](https://drive.google.com/drive/folders/1SHqsZqFqJ0EQksMrB6oh_k9O8biM0N3Q?usp=sharing).
