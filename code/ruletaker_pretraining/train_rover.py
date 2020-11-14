import os, sys, re
import argparse
from tqdm import tqdm
import numpy as np
import torch
from transformers import *
import subprocess
from pytorch_pretrained_bert import BertAdam
from utils_rover import load_dataset, load_dataset_fever, load_dataset_symmetric_fever
from sklearn.metrics import confusion_matrix

pretrained_model = "bert-base-cased"


def train(model, train_loader, dev_loader, optimizer, max_epochs):

    optimizer.zero_grad()

    best_dev_acc = 0
    best_epoch = 0
    subprocess.run("mkdir -p " + os.path.join(args.save_dir, "best"), shell=True)
    for epoch in range(max_epochs):
        print("starting epoch: %d" % epoch)
        epoch_loss = 0
        model.train()
        for _, batch in tqdm(enumerate(train_loader)):
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                token_type_ids=batch["token_type_ids"].to(model.device),
                labels=batch["labels"].to(model.device),
            )
            loss, logits = outputs[:2]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        print("train loss: %.4f" % epoch_loss)

        with torch.no_grad():
            model.eval()
            all_gold_labels, all_pred_labels = [], []
            for _, batch in tqdm(enumerate(dev_loader)):
                outputs = model(
                    input_ids=batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    token_type_ids=batch["token_type_ids"].to(model.device),
                )
                logits = outputs[0]
                pred_labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
                all_gold_labels.extend(list(np.array(batch["labels"])))
                all_pred_labels.extend(list(np.array(pred_labels)))

            all_gold_labels = np.array(all_gold_labels)
            all_pred_labels = np.array(all_pred_labels)
            dev_acc = (all_pred_labels == all_gold_labels).mean()
            print("dev acc: %.3f" % dev_acc)
            if dev_acc > best_dev_acc:
                print("new best acc. on dev: epoch %d acc %.3f" % (epoch, dev_acc))
                model.save_pretrained(os.path.join(args.save_dir, "best"))
                best_dev_acc = dev_acc
                best_epoch = epoch

        subprocess.run(
            "mkdir -p " + os.path.join(args.save_dir, "epoch" + str(epoch)), shell=True
        )
        model.save_pretrained(os.path.join(args.save_dir, "epoch" + str(epoch)))


def train_main(args):

    train_loader, num_train = load_dataset(
        os.path.join(args.input, "train.jsonl"), args.bs, shuffle=True
    )
    print("loaded train, # samples: %d" % num_train)
    dev_loader, num_dev = load_dataset(os.path.join(args.input, "dev.jsonl"), args.bs)
    print("loaded dev, # samples: %d" % num_dev)

    pretrained_model = BertForMultipleChoice.from_pretrained(args.checkpoint)
    pretrained_state_dict = pretrained_model.state_dict()
    bert_state_dict = {
        key: pretrained_state_dict[key]
        for key in pretrained_state_dict
        if not key.startswith("classifier")
    }
    # model = BertForSequenceClassification.from_pretrained(
    #     args.checkpoint, state_dict=bert_state_dict, num_labels=2
    # )
    model = BertForSequenceClassification.from_pretrained(
        args.checkpoint, state_dict=bert_state_dict, num_labels=3
    )
    print("loaded pretrained model")

    if torch.cuda.is_available():
        model = model.cuda()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    num_training_steps = int(len(train_loader) * args.max_epoch)
    optimizer = BertAdam(
        optimizer_grouped_parameters, lr=args.lr, warmup=0.1, t_total=num_training_steps
    )
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    # num_warmup_steps = 0.0
    # print(
    #     "total training steps: %d, warmup steps: %d"
    #     % (num_training_steps, num_warmup_steps)
    # )
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps,
    # )  # PyTorch scheduler

    train(
        model, train_loader, dev_loader, optimizer, args.max_epoch,
    )


def evaluate(model, data_loader):

    with torch.no_grad():
        model.eval()
        all_gold_labels, all_pred_labels = [], []
        for _, batch in tqdm(enumerate(data_loader)):
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                token_type_ids=batch["token_type_ids"].to(model.device),
            )
            logits = outputs[0]
            pred_labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
            all_gold_labels.extend(list(np.array(batch["labels"])))
            all_pred_labels.extend(list(np.array(pred_labels)))

        all_gold_labels = np.array(all_gold_labels)
        all_pred_labels = np.array(all_pred_labels)
        dev_acc = (all_pred_labels == all_gold_labels).mean()
        print("eval acc: %.3f" % dev_acc)
        print(confusion_matrix(all_gold_labels, all_pred_labels))
        # for pred, gold in zip(all_pred_labels, all_gold_labels):
        #     print("%s\t%s" % (pred, gold))


def eval_main(args):
    if args.task == "rover":
        data_loader, num_eval = load_dataset(args.input, args.bs, shuffle=False)
    elif args.task == "fever":
        data_loader, num_eval = load_dataset_fever(args.input, args.bs, shuffle=False)
    elif args.task == "fever_sym":
        data_loader, num_eval = load_dataset_symmetric_fever(
            args.input, args.bs, shuffle=False
        )
    print("loaded eval, # samples: %d" % num_eval)

    model = BertForSequenceClassification.from_pretrained(args.model)

    print("loaded pretrained model")
    if torch.cuda.is_available():
        model = model.cuda()

    evaluate(model, data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train rover model")
    subparsers = parser.add_subparsers(help="train, test")

    train_parser = subparsers.add_parser("train", help="train mode")
    train_parser.add_argument("-input", type=str, help="input dir")
    train_parser.add_argument(
        "-checkpoint",
        type=str,
        default="bert-base-cased",
        help="pretrained model, provide a custom RACE pretrained checkpoint",
    )
    train_parser.add_argument("-bs", type=int, default=8, help="batch size")
    train_parser.add_argument("-lr", type=float, default=3e-5, help="learning rate")
    train_parser.add_argument("-max-epoch", type=int, default=10, help="max epochs")
    train_parser.add_argument(
        "-save-dir", type=str, default="", help="save checkpoints"
    )
    train_parser.add_argument(
        "-weight-decay", type=float, default=0.0, help="weight decay"
    )
    train_parser.set_defaults(func=train_main)

    eval_parser = subparsers.add_parser("eval", help="eval mode")
    eval_parser.add_argument("-input", type=str, help="input file")
    eval_parser.add_argument("-model", type=str, help="trained model checkpoint")
    eval_parser.add_argument("-task", type=str, help="task name")
    eval_parser.add_argument("-bs", type=int, default=8, help="batch size")
    eval_parser.set_defaults(func=eval_main)

    args = parser.parse_args()
    args.func(args)
