import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, T5Tokenizer
from datasets import Dataset
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=512)

path_export = sys.argv[2]
random_seed = int(sys.argv[3])
test_size = float(sys.argv[4])
model_name = sys.argv[5]

early_stopping_patience = 10

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = T5Tokenizer.from_pretrained(model_name)

df = pd.read_csv(sys.argv[1])
df.columns = ["sequence", "label"]  # Ensure the column names match

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split dataset
dataset = dataset.train_test_split(test_size=test_size, seed=random_seed)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

# Apply tokenizer
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Remove unused columns
train_dataset = train_dataset.remove_columns(["sequence"])
eval_dataset = eval_dataset.remove_columns(["sequence"])

# Set format for PyTorch
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

for param in model.base_model.parameters():
    param.requires_grad = False

training_args = TrainingArguments(
    output_dir=f"{path_export}check_points",       
    evaluation_strategy="epoch",  
    save_strategy="epoch",
    learning_rate=5e-5,           
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=True,
    logging_dir=f"{path_export}logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2, 
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
print("Evaluation Results:", eval_results)

with open(f'{path_export}performances.json', 'w', encoding='utf-8') as f:
    dump(eval_results, f, ensure_ascii=False, indent=4)

name_model = model_name.split("/")[-1]

model.save_pretrained(f"{path_export}finetuned_{name_model}")
tokenizer.save_pretrained(f"{path_export}finetuned_{name_model}")

#del model
gc.collect()
torch.cuda.empty_cache()