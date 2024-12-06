import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from datasets import Dataset
import pandas as pd
from json import dump
import gc
from sklearn.utils import shuffle
import argparse
parser = argparse.ArgumentParser()

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

parser.add_argument(
    "-d", 
    "--dataset", 
    help="Input dataset to estimate physicochemical properties", 
    required=True)

parser.add_argument(
    "-s", 
    "--test_size", 
    help="Test size dataset", 
    type=float,
    default=0.2)

parser.add_argument(
    "-r", 
    "--random_seed", 
    help="Random seed", 
    type=int,
    default=42)

parser.add_argument(
    "-o", 
    "--path_output", 
    help="Path to save result", 
    required=True)

parser.add_argument(
    "-m", 
    "--model_name", 
    help="Name model to use", 
    required=True)

parser.add_argument(
    "-l", 
    "--max_lenght", 
    help="Max length of sequences", 
    type=int,
    default=1024)

parser.add_argument(
    "-e", 
    "--early_stop", 
    help="Early stop tolerance", 
    type=int,
    default=10)

args = parser.parse_args()

df_data = pd.read_csv(args.dataset)
test_size = args.test_size
random_seed = args.random_seed
path_export = args.path_output
model_name = f"RaphaelMourad/{args.model_name}"
max_length = args.max_lenght
early_stopping_patience = args.early_stop

df_pos = df_data[df_data["label"] == 1]
df_neg = df_data[df_data["label"] == 0]
df_neg = shuffle(df_neg, n_samples=len(df_pos), random_state=random_seed)
df_data = pd.concat([df_pos, df_neg], axis=0)

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
    output_dir=f"{path_export}explored_checkpoints/{name_model}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=30,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir=f"{path_export}logs/{name_model}",
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

with open(f'{path_export}performances_{name_model}.json', 'w', encoding='utf-8') as f:
    dump(eval_results, f, ensure_ascii=False, indent=4)

model.save_pretrained(f"{path_export}finetuned_{name_model}")
tokenizer.save_pretrained(f"{path_export}finetuned_{name_model}")

#del model
gc.collect()
torch.cuda.empty_cache()