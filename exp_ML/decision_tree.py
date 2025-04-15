import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import time
import os
import psutil


def decision_tree(trainPath, testPath, outputPath):
    data = pd.read_csv(trainPath)
    data_test = pd.read_csv(testPath)
    data_1 = data["heartbeat_signals"].str.split(",", expand=True)
    data_test_1 = data_test["heartbeat_signals"].str.split(",", expand=True)
    data_2 = np.array(data_1).astype("float32").reshape(-1, 100)
    data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100)

    x_train = torch.tensor(data_2)
    x_test = torch.tensor(data_test_2)
    y_train = torch.tensor(data.label, dtype=int)
    y_test = torch.tensor(data_test.answer, dtype=int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    x_train_cpu = x_train.cpu().numpy()
    y_train_cpu = y_train.cpu().numpy()
    x_test_cpu = x_test.cpu().numpy()
    y_test_cpu = y_test.cpu().numpy()

    # 初始化决策树分类器
    clf = DecisionTreeClassifier()

    # 训练模型
    clf.fit(x_train_cpu, y_train_cpu)

    # 进行预测
    y_pred = clf.predict(x_test_cpu)

    # 评估模型
    # print("Accuracy:", accuracy_score(y_test_cpu, y_pred))
    # print(classification_report(y_test_cpu, y_pred))
    with open(outputPath, 'w') as file:
        file.write(f"Decision_tree\n")
        file.write(f"Accuracy: {accuracy_score(y_test_cpu, y_pred)}\n")
        file.write(classification_report(y_test_cpu, y_pred))
        file.write(f"\n")
