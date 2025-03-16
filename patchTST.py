from transformers import PatchTSTConfig, PatchTSTForClassification
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from argparse import Namespace
import yaml
import wandb
from sklearn.impute import SimpleImputer
from models import LSTM4TS
# os.environ["WANDB_MODE"] = "offline"


def load_args(file_path):
    with open(file_path, 'r') as f:
        args = yaml.safe_load(f)
    args = Namespace(**args)
    return args


parser = argparse.ArgumentParser(
    description='Load parameters from a YAML file.')
parser.add_argument('--config', type=str, default='config/bay_point.yaml')
parameters = parser.parse_args()

# load args
global args
args = load_args(parameters.config)
print(args)
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


data = pd.read_csv(train_dataset_name)
data_test = pd.read_csv(test_dataset_name)

data.head()
data_1 = data["heartbeat_signals"].str.split(",", expand=True)
data_test_1 = data_test["heartbeat_signals"].str.split(",", expand=True)
np.array(data.label)
# data_2 = np.array(data_1).astype("float32").reshape(-1, 100, 1)
# data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100, 1)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_2 = np.array(data_1).astype("float32").reshape(-1, 100)
data_2 = imputer.fit_transform(data_2)
data_2 = data_2.reshape(-1, 100, 1)
data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100)
data_test_2 = imputer.fit_transform(data_test_2)
data_test_2 = data_test_2.reshape(-1, 100, 1)

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
train_loader = DataLoader(dataset, batch_size=128,
                          shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset_test, batch_size=128,
                         shuffle=True, pin_memory=False)

# classification task with two input channel2 and 3 classes
config = PatchTSTConfig(
    num_input_channels=1,
    num_targets=args.label,
    context_length=100,
    patch_length=50,
    stride=50,
    use_cls_token=True,
)
# model = PatchTSTForClassification(config=config)
model = LSTM4TS(input_dim=1, seq_len=100, num_classes=args.label, lstm_layers=2, hidden_size=256)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# model train
num_epochs = 200

# with open('training_results.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # 写入标题行
#     csvwriter.writerow(
#         ['Epoch', 'Train Loss', 'Train ACC', 'Test Loss', 'Test ACC', 'Best Test Loss', 'Best Test ACC'])

best_test_loss = float('inf')
best_test_acc = 0.0
best_train_loss = float('inf')
best_train_acc = 0.0
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs
        outputs = model(inputs)
        # probabilities = torch.nn.functional.softmax(outputs.prediction_logits, dim=-1)

        # loss = criterion(outputs.prediction_logits, labels)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        # 获取最大概率的预测类别
        # _, predicted = torch.max(outputs.prediction_logits, 1)
        _, predicted = torch.max(outputs, 1)
        # 更新正确预测的数量
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
    epoch_loss = loss.item()
    train_acc = correct_predictions / total_predictions

    if epoch_loss < best_train_loss:
        best_train_loss = epoch_loss

    if train_acc > best_train_acc:
        best_train_acc = train_acc

    res_train = {'train/loss': epoch_loss, 'train/acc': train_acc,
                 'best/train_loss': best_train_loss, 'best/train_acc': best_train_acc}

    # 测试阶段
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            # test_loss += criterion(outputs.prediction_logits, labels).item()
            test_loss += criterion(outputs, labels).item()
            # 获取最大概率的预测类别
            # _, predicted = torch.max(outputs.prediction_logits, 1)
            _, predicted = torch.max(outputs, 1)
            # 更新正确预测的数量
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct_predictions / total_predictions

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = model.state_dict()

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_state = model.state_dict()

    res_test = {'test/loss': test_loss, 'test/acc': test_acc,
                'best/test_loss': best_test_loss, 'best/test_acc': best_test_acc}

    # print(f"best/test_loss {best_test_loss}, best/test_acc: {best_test_acc}")
    res = {**res_train, **res_test}
    if args.wandb:
        wandb.log(res)

torch.save(best_model_state, model_name)

# model test
# model = PatchTSTForClassification(config=config)
model = LSTM4TS(input_dim=1, seq_len=100, num_classes=args.label, lstm_layers=2, hidden_size=256)
model.to(device)
model.load_state_dict(torch.load(model_name))
model.eval()

data_test = pd.read_csv(test_label_name)

data_test_1 = data_test["heartbeat_signals"].str.split(",", expand=True)
ids = data_test["id"].values
# data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100, 1)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100)
data_test_2 = imputer.fit_transform(data_test_2)
data_test_2 = data_test_2.reshape(-1, 100, 1)

torch.set_printoptions(precision=7)

x_test = torch.tensor(data_test_2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

x_test = x_test.to(device)
batch_size = 128

num_samples = len(x_test)
num_batches = (num_samples + batch_size - 1) // batch_size

all_outputs = []


with torch.no_grad():
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        inputs_batch = x_test[start_idx:end_idx]
        outputs_batch = model(inputs_batch)
        # all_outputs.append(outputs_batch.prediction_logits)
        all_outputs.append(outputs_batch)

all_outputs_tensor = torch.cat(all_outputs, dim=0)
normalized_outputs = F.softmax(all_outputs_tensor, dim=1)
predicted_labels_cpu = normalized_outputs.cpu()

predicted_labels_np = predicted_labels_cpu.numpy()
torch.set_printoptions(precision=8)
numpy_array_1 = pd.DataFrame(predicted_labels_np)
if args.label == 5:
    numpy_array_1.columns = ["label_0", "label_1",
                             "label_2", "label_3", "label_4"]
elif args.label == 4:
    numpy_array_1.columns = ["label_0", "label_1",
                             "label_2", "label_3"]
elif args.label == 3:
    numpy_array_1.columns = ["label_0", "label_1",
                             "label_2"]
numpy_array_1.index = ids
numpy_array_1.index.name = "id"
csv_name = answer_csv_name
numpy_array_1.to_csv(csv_name, index=True)

df_predictions = pd.read_csv(csv_name)
df_answers = pd.read_csv(test_dataset_name)

df_predictions['id'] = pd.to_numeric(
    df_predictions['id'], errors='coerce').dropna().astype(int)
df_answers['id'] = pd.to_numeric(
    df_answers['id'], errors='coerce').dropna().astype(int)

df_predictions['Predicted_Label'] = df_predictions.iloc[:, 1:].idxmax(
    axis=1).apply(lambda x: int(x.lstrip('label_')))

df_answers['answer'] = pd.to_numeric(
    df_answers['answer'], errors='coerce').dropna().astype(int)

merged_df = pd.merge(df_predictions[['id', 'Predicted_Label']], df_answers[[
                     'id', 'answer']], on='id', how='left')

accuracy = merged_df[merged_df['answer'].notna()]['Predicted_Label'].eq(
    merged_df['answer']).mean()

print(f"accuracy: {accuracy:.4f}")

# 假设 'answer' 是实际类别列，'Predicted_Label' 是预测类别列
accuracy_by_class = (merged_df['Predicted_Label'] == merged_df['answer'])
accuracy_by_class = accuracy_by_class.groupby(merged_df['answer']).mean()

# 打印每一类的准确率
print(accuracy_by_class)

result_df = df_predictions[['id', 'Predicted_Label']]

result_df.to_csv(answer_label_name, index=False)

print("Predicted labels have been saved")
