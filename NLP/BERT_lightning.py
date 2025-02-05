import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import pandas as pd
import argparse
import lightning as L
from lightning.pytorch import loggers as pl_loggers
import torchmetrics

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.model_selection import train_test_split


class BERTClassifier(L.LightningModule):
    def __init__(self, n_classes: int, weight_decay, steps_per_epoch=None, n_epochs=None, lr=2e-5):
        super().__init__()
        #self.save_hyperparameters()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr

        self.weight_decay = weight_decay

        self.train_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=2)
        self.test_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=2)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        #inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['label']}
        #outputs = self(**inputs)
        #print(inputs)
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        #print(outputs)
        loss = outputs.loss
        self.train_acc(outputs.logits, batch['label'])
        self.log_dict({'train_loss': loss, 'train_acc': self.train_acc}, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        frob_norm = frobenius_norm(self)
        self.log_dict({'frob_norm': frob_norm, 'margin': 1 / frob_norm})

    def on_train_end(self):
        frobenius_norm_value = frobenius_norm(self)
        print(f"Frobenius Norm of the model after training: {frobenius_norm_value:.4f}")

    """
    def validation_step(self, batch, batch_idx):
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['labels']}
        outputs = self(**inputs)
        val_loss = outputs.loss
        return val_loss
    """

    def test_step(self, batch, batch_idx):
        #inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['label']}
        #outputs = self(**inputs)
        #outputs = self(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        test_loss = outputs.loss
        self.test_acc(outputs.logits, batch['label'])
        self.log_dict({'test_loss': test_loss, 'test_acc': self.test_acc}, on_step=False, on_epoch=True)
        return test_loss

    def predict(self, texts):
        self.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        with torch.no_grad():
            logits = self(input_ids, attention_mask).logits
        return torch.argmax(logits, dim=1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


# Function to calculate Frobenius norm of model parameters
def frobenius_norm(model):
    norm = 0
    for param in model.parameters():
        norm += torch.norm(param, p="fro") ** 2
    return torch.sqrt(norm).item()


class IMDbDataModule(L.LightningDataModule):
    def __init__(self, tokenizer_name='bert-base-uncased', max_length=128, batch_size=16):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)

    def prepare_data(self):
        load_dataset('imdb')

    def setup(self, stage=None):
        dataset = load_dataset('imdb')

        # Tokenization function
        #def tokenize_data(example):
         #   return self.tokenizer(example['text'], padding='max_length', truncation=True, max_length=self.max_length)

        #tokenized_datasets = dataset.map(tokenize_data, batched=True)

        tokenized_dataset = dataset.map(lambda sample: self.tokenizer(sample['text'], padding='max_length', truncation=True, max_length=self.max_length), batched=True)
        tokenized_dataset.set_format(type='torch')

        #print(tokenized_dataset['train'])

        if stage == "fit" or stage is None:
            self.train_data = tokenized_dataset['train']
            #self.train_data, self.val_data = random_split(train_data, [0.9, 0.1])
            #self.train_data, self.val_data = train_test_split(train_data, test_size=0.1, random_state=42)

        if stage == "test" or stage is None:
            self.test_data = tokenized_dataset['test']

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    #def val_dataloader(self):
     #   return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


# Run the experiment several times
def run_experiments(
    num_runs=1,
    output_file="results.csv",
    weight_decay=0.0,
    num_epochs=5,
    batch_size=16,
    max_length=128,
    learning_rate=2e-5,
):
    for run in range(num_runs):
        imdb = IMDbDataModule(max_length=max_length, batch_size=batch_size)
        model = BERTClassifier(n_classes=2, weight_decay=weight_decay, lr=learning_rate)

        trainer = L.Trainer(accelerator='gpu', max_epochs=num_epochs, default_root_dir=output_file.split('.csv')[0], enable_checkpointing=False)
        trainer.fit(model=model, datamodule=imdb)

        trainer.test(model=model, datamodule=imdb)

        # Use the fine-tuned model for prediction
        texts = ["This movie is fantastic!", "I didn't like this movie at all."]
        print(f'Predict following examples: {texts}')
        predictions = model.predict(texts)

        # Map predictions back to labels
        label_map = {0: 'Negative', 1: 'Positive'}
        predicted_labels = [label_map[pred.item()] for pred in predictions]
        print(predicted_labels)

        print(f"Run {run+1}/{num_runs} completed.")


# Run the main function with a specific weight decay
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--number_of_runs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results_hyper_3.csv")
    args = parser.parse_args()

    run_experiments(
        num_runs=args.number_of_runs,
        output_file=args.output_dir,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
