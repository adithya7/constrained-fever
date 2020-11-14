# Constrained Fact Verification

This repository contains the code, data and models corresponding to the EMNLP 2020 paper, [Constrained Fact Verification for FEVER](https://www.aclweb.org/anthology/2020.emnlp-main.629).

## Datasets

To download the RuleTaker-CWA, RuleTaker-Skip-fact and anonymized FEVER datasets, please refer to the [link](https://drive.google.com/drive/folders/1dznaQruOezoxrsFZKiC1V-vF25FnSEAG?usp=sharing).

Alternatively, to prepare these datasets from scratch, refer to [code/build_datasets](code/build_datasets).

## Pretrained models

To reproduce the results presented in the paper, first download the checkpoints as below,

```bash
cd checkpoints
python download_checkpoints.py
cd ..
```

Download the three evaluation sets, Standard FEVER, Symmetric FEVER and Anonymized FEVER from the [link](https://drive.google.com/drive/folders/1zthG6CS2HteYyj59gmrIthblG0ovSkzd?usp=sharing).

For evaluating the FEVER models, please refer to [code/bert-concat](code/bert-concat). To evaluate the pretrained RuleTaker-CWA and RuleTaker-Skip-fact models, refer to [code/ruletaker_pretraining](code/ruletaker_pretraining).

Alternatively, if you wish to train new models or experiment further, please follow the below steps.

## Training

### RuleTaker pretraining

This involves two steps,

1. Pretraining on RACE ([Lai et al. 2017](https://www.aclweb.org/anthology/D17-1082/)) (refer to [code/race_pretraining](code/race_pretraining))
2. Further fine-tune on the RuleTaker dataset. (refer to [code/ruletaker_pretraining](code/ruletaker_pretraining)).

### FEVER finetuning

After the above RuleTaker pretraining, we experiment with three networks for training on FEVER. We use the fine-tuned BERT weights from previous step to initialize the encoder.

1. BERT-concat (refer to [code/bert-concat](code/bert-concat)).
2. KGAT ([Liu et al. 2020](https://www.aclweb.org/anthology/2020.acl-main.655/)) (refer to [code/kgat](code/kgat)).
3. Transformer-XH ([Zhao et al. 2020](https://openreview.net/forum?id=r1eIiCNYwS)) (refer to [code/transformer-xh](code/transformer-xh)).

Alternatively, you can use the fine-tuned BERT weights from previous step (RuleTaker pretraining) to further train on any other fact verification dataset of your choice.

## Other Resources

- For more information about the FEVER 1.0 shared task, please refer to [fever.ai](http://fever.ai).
- For more information about the original RuleTaker dataset (Clark et al. 2020), please refer to [RuleTaker](https://rule-reasoning.apps.allenai.org).
- To experiment with other transformer-based encoders like RoBERTa, checkout [huggingface](https://huggingface.co).

## Citation

If you find these resources helpful in your research, consider citing the paper,

```BibTeX
@inproceedings{pratapa-etal-2020-constrained,
    title = "{C}onstrained {F}act {V}erification for {FEVER}",
    author = "Pratapa, Adithya and Jayanthi, Sai Muralidhar and Nerella, Kavya",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.629",
    pages = "7826--7832",
}
```

For any questions/issues, please feel free to create an issue.
