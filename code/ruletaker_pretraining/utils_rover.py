import os, sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import *

pretrained_model = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
MAX_LEN = 512
fever2id = {"NEI": 0, "REFUTES": 1, "SUPPORTS": 2}
id2fever = {0: "NEI", 1: "REFUTES", 2: "SUPPORTS"}


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
        batch_samples.append([sample["context"], sample["text"]])
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
    with open(file_path, "r") as rf:
        for line in rf:
            json_dict = json.loads(line.strip())
            context = json_dict["context"]
            for question in json_dict["questions"]:
                text = question["text"]
                label = question["label"]
                data.append({"context": context, "text": text})
                labels.append(fever2id[label])

    dataset = customDataset(data, labels)
    data_loader = DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=4,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
    )
    return data_loader, len(data)


def load_dataset_fever(file_path, batch_size, shuffle=False):
    data = []
    labels = []
    with open(file_path, "r") as rf:
        for line in rf:
            json_dict = json.loads(line.strip())
            label = False
            if json_dict["label"] == "SUPPORTS":
                label = True
            elif (
                json_dict["label"] == "REFUTES"
                or json_dict["label"] == "NOT ENOUGH INFO"
            ):
                label = False
            else:
                continue
            labels.append(label)
            claim = json_dict["claim"]
            evidence = ""
            for evd_sent in json_dict["ranked_evidence"]:
                wikiTitle, sent = evd_sent[0], evd_sent[2]
                # evidence += " ".join(
                #     [wikiTitle, sent]
                # )  # concat wikititle and evidence sentence
                evidence += sent  # just evidence sentence
                evidence += " "
            evidence = evidence.rstrip()
            data.append({"context": evidence, "text": claim})

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
    data = []
    labels = []
    with open(file_path, "r") as rf:
        for line in rf:
            json_dict = json.loads(line.strip())
            label = None
            assert (
                json_dict["gold_label"] == json_dict["label"]
            ), "two labels in the input"
            if json_dict["label"] == "SUPPORTS":
                label = True
            elif json_dict["label"] == "REFUTES":
                label = False
            assert label is not None, "unexpected label"
            labels.append(label)
            claim = json_dict["claim"]
            evidence = json_dict["evidence"]
            data.append({"context": evidence, "text": claim})

    dataset = customDataset(data, labels)
    data_loader = DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=4,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
    )
    return data_loader, len(data)
