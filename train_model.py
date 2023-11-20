import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7"

import torch
import datasets
import transformers
from transformers import TrainingArguments, Trainer

def top_n_accuracy(preds: torch.Tensor, labels: torch.Tensor, n: int = 1) -> float:
    assert preds.shape[0] == labels.shape[0]
    batch_size, n_classes, *extra_dims = preds.shape
    _, indices = torch.topk(preds, n, dim=1)  # of shape (batch_size, n) or (batch_size, n, d_1, ..., d_n)
    pattern = " ".join([f"d_{i+1}" for i in range(len(extra_dims))])
    pattern = f"batch_size {pattern} -> batch_size n {pattern}"
    correct = torch.sum(indices == einops.repeat(labels, pattern, n=n))
    acc = correct.item() / preds.shape[0]
    for d in extra_dims:
        acc = acc / d
    return acc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return top_n_accuracy(logits, labels, n=1)


def main():
    refined_classes = ["资讯","财经","体育","时政","娱乐","社会","科技","汽车","健康","萌宠","国际","生活", "美食", "游戏"]

    path = "/home/wenhao/Jupyter/wenhao/workspace/AI-Testing/text/data/roberta-base-finetuned-ifeng-chinese"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)

    model.classifier = torch.nn.Linear(768, 14, bias=True)

    model.config.id2label = {i: label for i, label in enumerate(refined_classes)}
    model.num_labels = len(refined_classes)

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
