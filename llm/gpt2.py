from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTConfig, PatchTSTForClassification
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from transformers import GPT2Model
import torch.nn as nn
import wandb
import argparse
from argparse import Namespace
import yaml


class GPT4TS(nn.Module):
    def __init__(self, input_dim=1, seq_len=100, num_classes=4, device="cuda:0", gpt_layers=6):
        super(GPT4TS, self).__init__()

        # GPT2 Model
        self.gpt2 = GPT2Model.from_pretrained(
            "./llm/gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        print("GPT2 Loaded with {} layers".format(gpt_layers))

        for name, param in self.gpt2.named_parameters():
            # param.requires_grad = True
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.embedding_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=self.gpt2.config.n_embd,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=self.gpt2.config.n_embd, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256*2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = self.embedding_layer(x)  
        x = x.permute(0, 2, 1)  
        gpt2_output = self.gpt2(inputs_embeds=x).last_hidden_state
        lstm_output, _ = self.lstm(gpt2_output)
        cls_embedding = lstm_output[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits

def load_args(file_path):
    with open(file_path, 'r') as f:
        args = yaml.safe_load(f)
    args = Namespace(**args)
    return args


parser = argparse.ArgumentParser(
    description='Load parameters from a YAML file.')
parser.add_argument('--config', type=str, default='config/bay_point.yaml')
parser.add_argument('--epochs', type=int, default=2000)
parameters = parser.parse_args()

# load args
global args
args = load_args(parameters.config)

if args.wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        entity='22373442',
        project="LLM_signal",
        name=args.name,
        reinit=True
    )

train_dataset_name = args.train_dataset_name
test_dataset_name = args.test_dataset_name
test_label_name = args.test_label_name

answer_label_name = args.answer_label_name
model_name = args.model_name
answer_csv_name = args.answer_csv_name
label_num = args.label

data = pd.read_csv(train_dataset_name)
data_test = pd.read_csv(test_dataset_name)

data.head()
data_1 = data["heartbeat_signals"].str.split(",", expand=True)
data_test_1 = data_test["heartbeat_signals"].str.split(",", expand=True)
np.array(data.label)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_2 = np.array(data_1).astype("float32").reshape(-1, 100)
data_2 = imputer.fit_transform(data_2)
# data_2 = data_2.reshape(-1, 100, 1)
data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100)
data_test_2 = imputer.fit_transform(data_test_2)
# data_test_2 = data_test_2.reshape(-1, 100, 1)

scaler = StandardScaler()
data_2 = scaler.fit_transform(data_2.reshape(-1, 100)).reshape(-1, 100, 1)
data_test_2 = scaler.transform(
    data_test_2.reshape(-1, 100)).reshape(-1, 100, 1)

torch.set_printoptions(precision=7)

x_train = torch.tensor(data_2)
x_test = torch.tensor(data_test_2)
y_train = torch.tensor(data.label, dtype=int)
y_test = torch.tensor(data_test.answer, dtype=int)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)
dataset = TensorDataset(x_train, y_train)
dataset_test = TensorDataset(x_test, y_test)
train_loader = DataLoader(dataset, batch_size=32,
                          shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset_test, batch_size=32,
                         shuffle=True, pin_memory=False)
# Replace PatchTST with GPT4TS
input_dim = data_2.shape[-1]
seq_len = data_2.shape[1]
gpt_model = GPT4TS(input_dim=1, seq_len=seq_len, num_classes=args.label,
                   device=device, gpt_layers=3).to(device)
gpt_model.to(device)

# Update model definition
model = gpt_model
model.to(device)

# Adjust training loop for GPT4TS
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = parameters.epochs
best_test_loss = float('inf')
best_test_acc = 0.0
best_train_loss = float('inf')
best_train_acc = 0.0
# Update training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accuracy
        predicted = torch.argmax(outputs, dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    train_acc = correct_predictions / total_predictions
    # scheduler.step()
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}")
    if epoch_loss < best_train_loss:
        best_train_loss = epoch_loss

    if train_acc > best_train_acc:
        best_train_acc = train_acc

    res_train = {'train/loss': epoch_loss, 'train/acc': train_acc,
                 'best/train_loss': best_train_loss, 'best/train_acc': best_train_acc}

    # test
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, labels)
            predicted = torch.argmax(outputs, dim=1)

            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct_predictions / total_predictions
    print(
        f"test_loss: {test_loss:.4f}, test_accuracy: {test_acc:.4f}")
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        # best_model_state = model.state_dict()

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_state = model.state_dict()

    res_test = {'test/loss': test_loss, 'test/acc': test_acc,
                'best/test_loss': best_test_loss, 'best/test_acc': best_test_acc}

    print(f"best/test_loss {best_test_loss}, best/test_acc: {best_test_acc}")
    res = {**res_train, **res_test}
    if args.wandb:
        wandb.log(res)

# Save the model
torch.save(model.state_dict(), model_name)
