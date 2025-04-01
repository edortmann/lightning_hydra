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

from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from datetime import datetime


# -------------------------
# Lightning Module for BERT
# -------------------------
class BERTClassifier(L.LightningModule):
    def __init__(self, n_classes: int, weight_decay, results, steps_per_epoch=None, n_epochs=None, lr=2e-5):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay

        # tracked parameters after model has been trained
        self.frobenius_norm_value = None
        self.train_acc_final = None
        self.test_acc_final = None
        self.results = results

        # Freeze all parameters first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 encoder layers, the pooler, and the classifier
        for name, param in self.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name or "classifier" in name:
                param.requires_grad = True

        self.train_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=n_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=n_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        loss = outputs.loss
        self.train_acc(outputs.logits, batch['label'])
        self.log_dict({'train_loss': loss, 'train_acc': self.train_acc}, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_end(self):
        frobenius_norm_value = frobenius_norm(self)
        self.frobenius_norm_value = frobenius_norm_value
        print(f"Frobenius Norm of the model after training: {frobenius_norm_value:.4f}")

        # Access the train accuracy from the trainer's callback_metrics
        train_acc = self.trainer.callback_metrics.get("train_acc")
        self.train_acc_final = train_acc.item()
        print(f"Final train accuracy: {train_acc.item()}")

    def test_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        test_loss = outputs.loss
        self.test_acc(outputs.logits, batch['label'])
        self.log_dict({'test_loss': test_loss, 'test_acc': self.test_acc}, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def on_test_end(self):
        # Access the test accuracy from the trainer's callback_metrics
        test_acc = self.trainer.callback_metrics.get("test_acc")
        self.test_acc_final = test_acc.item()
        print(f"Final test accuracy: {test_acc.item()}")

        self.results.append(
            {
                "train_accuracy": self.train_acc_final,
                "test_accuracy": self.test_acc_final,
                "frobenius_norm": self.frobenius_norm_value,
                "margin": 1 / self.frobenius_norm_value,
                "train-test acc": self.train_acc_final - self.test_acc_final,
                "weight_decay": self.weight_decay,
            }
        )

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
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer

    def get_results(self):
        return self.results


# -------------------------
# Function to calculate Frobenius norm of model parameters
# -------------------------
def frobenius_norm(model):
    norm = 0
    for param in model.parameters():
        if param.requires_grad:
            norm += torch.norm(param, p="fro") ** 2
    return torch.sqrt(norm).item()


# -------------------------
# Data Module for Imdb
# -------------------------
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

        tokenized_dataset = dataset.map(lambda sample: self.tokenizer(sample['text'], padding='max_length', truncation=True, max_length=self.max_length), batched=True)
        tokenized_dataset.set_format(type='torch')

        if stage == "fit" or stage is None:
            self.train_data = tokenized_dataset['train']

        if stage == "test" or stage is None:
            self.test_data = tokenized_dataset['test']

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=11)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


# Run the experiment several times
def run_experiments(
    output_file="final_results.csv",
    weight_decay=0.01,
    num_epochs=5,
    batch_size=16,
    max_length=128,
    learning_rate=2e-5,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    weight_decays = [weight_decay]

    imdb = IMDbDataModule(max_length=max_length, batch_size=batch_size)
    results = []

    for weight_decay in weight_decays:

        model = BERTClassifier(n_classes=2, weight_decay=weight_decay, lr=learning_rate, results=results)

        trainer = L.Trainer(accelerator='gpu', max_epochs=num_epochs, default_root_dir=output_file.split('.csv')[0], enable_checkpointing=False)
        trainer.fit(model=model, datamodule=imdb)

        trainer.test(model=model, datamodule=imdb)

        results = model.get_results()

        # Use the fine-tuned model for prediction
        texts = ["This movie is fantastic!", "I didn't like this movie at all."]
        print(f'Predict following examples: {texts}')
        predictions = model.predict(texts)

        # Map predictions back to labels
        label_map = {0: 'Negative', 1: 'Positive'}
        predicted_labels = [label_map[pred.item()] for pred in predictions]
        print(predicted_labels)

        print(f"Run with weight decay {weight_decay} completed.")

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Results saved to final_results.csv")


# Run the main function with a specific weight decay
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--number_of_runs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results.csv")
    args = parser.parse_args()

    run_experiments(
        output_file=args.output_dir,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
