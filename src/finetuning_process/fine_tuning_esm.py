import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from datasets import Dataset
import pandas as pd
import sys
from json import dump
import gc

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def tokenize_function(examples):
    return tokenizer(
        examples["sequence"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

max_length = 50
early_stopping_patience = 10

df_data = pd.read_csv(sys.argv[1])
test_size = float(sys.argv[2])
random_seed = int(sys.argv[3])
path_export = sys.argv[4]
model_name = sys.argv[5]

name_model = model_name.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

df_data.columns = ["sequence", "label"]

dataset = Dataset.from_pandas(df_data)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_test_split = tokenized_datasets.train_test_split(test_size=test_size, seed=random_seed)

train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

training_args = TrainingArguments(
    output_dir=f"{path_export}explored_checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=30,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir=f"{path_export}logs",
    logging_steps=10,
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
)

trainer.train()

eval_results = trainer.evaluate()

with open(f'{path_export}performances.json', 'w', encoding='utf-8') as f:
    dump(eval_results, f, ensure_ascii=False, indent=4)

model.save_pretrained(f"{path_export}finetuned_{name_model}")
tokenizer.save_pretrained(f"{path_export}finetuned_{name_model}")

#del model
gc.collect()
torch.cuda.empty_cache()