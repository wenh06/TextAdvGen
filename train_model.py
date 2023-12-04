import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import datasets
import einops
import numpy as np
import torch
import transformers
from transformers import TrainingArguments, Trainer, EvalPrediction


def top_n_accuracy(preds: torch.Tensor, labels: torch.Tensor, n: int = 1) -> torch.Tensor:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    assert preds.shape[0] == labels.shape[0]
    batch_size, n_classes, *extra_dims = preds.shape
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    _, indices = torch.topk(preds, n, dim=1)  # of shape (batch_size, n) or (batch_size, n, d_1, ..., d_n)
    pattern = " ".join([f"d_{i+1}" for i in range(len(extra_dims))])
    pattern = f"batch_size {pattern} -> batch_size n {pattern}"
    correct = torch.sum(indices == einops.repeat(labels, pattern, n=n))
    acc = correct / preds.shape[0]
    for d in extra_dims:
        acc = acc / d
    return acc


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred  # numpy.ndarray, numpy.ndarray
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    accuracies = {
        f"top{k}-accuracy": top_n_accuracy(logits, labels, n=k) for k in (1, 3, 5)
    }
    return accuracies


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def main():
    refined_classes = ["资讯","财经","体育","时政","娱乐","社会","科技","汽车","健康","萌宠","国际","生活", "美食", "游戏"]

    path = "/home/wenhao/Jupyter/wenhao/workspace/AI-Testing/text/data/roberta-base-finetuned-ifeng-chinese"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)

    model.classifier = torch.nn.Linear(768, 14, bias=True)

    model.config.id2label = {i: label for i, label in enumerate(refined_classes)}
    model.num_labels = len(refined_classes)

    freeze_backbone(model)

    dataset = datasets.load_dataset(
        "csv",
        data_files={
            "train": ["/home/wenhao/Jupyter/wenhao/workspace/AI-Testing/text/data/new_data/textData_hf_train.csv.gz"],
            "test": "/home/wenhao/Jupyter/wenhao/workspace/AI-Testing/text/data/new_data/textData_hf_test.csv.gz",
        },
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="/home/wenhao/Jupyter/wenhao/workspace/AI-Testing/test_trainer",
        per_device_train_batch_size=4,
        evaluation_strategy="epoch",
        num_train_epochs=30,
        save_total_limit=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save model
    trainer.save_model("/home/wenhao/Jupyter/wenhao/workspace/AI-Testing/test_trainer")


if __name__ == "__main__":
    main()

    # nohup python -u /home/wenhao/Jupyter/wenhao/workspace/AI-Testing/text/data/TextAdvGen/train_model.py > /home/wenhao/Jupyter/wenhao/workspace/AI-Testing/tmp/train.log 2>&1 & echo $! > /home/wenhao/Jupyter/wenhao/workspace/AI-Testing/tmp/train.pid
