import os, sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import *

pretrained_model = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
MAX_LEN = 256
fever2id = {"NOT ENOUGH INFO": 0, "REFUTES": 1, "SUPPORTS": 2}
id2fever = {0: "NOT ENOUGH INFO", 1: "REFUTES", 2: "SUPPORTS"}


class customDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def custom_collate_fn(batch):
    batch_samples = []
    batch_labels = []
    for sample, label in batch:
        batch_samples.append(sample)
        batch_labels.append(label)
    batch_tokens = tokenizer.batch_encode_plus(
        batch_samples,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_tensors="pt",
        add_special_tokens=True,
        truncation_strategy="longest_first",
    )
    batch_labels = torch.tensor(batch_labels, dtype=torch.int64)
    out_dict = {
        "input_ids": batch_tokens["input_ids"],
        "attention_mask": batch_tokens["attention_mask"],
        "token_type_ids": batch_tokens["token_type_ids"],
        "labels": batch_labels,
    }
    return out_dict


def load_dataset(file_path, batch_size, shuffle=False):
    data = []
    labels = []
    special_tokens = {
        "additional_special_tokens": ["ent0", "ent1", "ent2", "ent3", "ent4"]
    }
    tokenizer.add_special_tokens(special_tokens)
    with open(file_path, "r") as rf:
        for line in rf:
            json_dict = json.loads(line.strip())
            claim = json_dict["claim"]
            if "ranked_evidence" in json_dict:
                evidences = json_dict["ranked_evidence"]  # to eval on std. set
            else:
                evidences = json_dict["evidence"]  # to eval on anon. and symm. sets
            concat_evd = []
            for evd in evidences:
                wikiTitle = evd[0]
                wikiTitle = wikiTitle.replace("_", " ")
                sent = evd[2]
                concat_evd.append(wikiTitle)
                concat_evd.append(sent)
            data_instance = [claim] + [" ".join(concat_evd)]
            data.append(data_instance)
            labels.append(fever2id[json_dict["label"]])

    dataset = customDataset(data, labels)
    data_loader = DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=4,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
    )
    return data_loader, len(data)


def load_dataset_symmetric_fever(file_path, batch_size, shuffle=False):
    return NotImplementedError
