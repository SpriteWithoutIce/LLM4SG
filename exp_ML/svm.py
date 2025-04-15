import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import time
import os
import psutil


def SVM(trainPath, testPath, outputPath):
    data = pd.read_csv(trainPath)
    data_test = pd.read_csv(testPath)
    data.head()
    data_1 = data["heartbeat_signals"].str.split(",", expand=True)

    data_test_1 = data_test["heartbeat_signals"].str.split(",", expand=True)
    np.array(data.label)
    data_2 = np.array(data_1).astype("float32").reshape(-1, 100)
    data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_2 = np.array(data_1).astype("float32").reshape(-1, 100)
    data_2 = imputer.fit_transform(data_2)
    data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100)
    data_test_2 = imputer.fit_transform(data_test_2)
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

    # 确保数据在CPU上
    x_train_cpu = x_train.cpu().numpy()
    y_train_cpu = y_train.cpu().numpy()
    x_test_cpu = x_test.cpu().numpy()
    y_test_cpu = y_test.cpu().numpy()

    # 初始化SVM分类器
    clf = svm.SVC(kernel='linear', C=1.0)

    # 训练模型
    clf.fit(x_train_cpu, y_train_cpu)

    # 进行预测
    y_pred = clf.predict(x_test_cpu)

    # 评估模型
    # print("Accuracy:", accuracy_score(y_test_cpu, y_pred))
    # print(classification_report(y_test_cpu, y_pred))
    with open(outputPath, 'a') as file:
        file.write(f"SVM\n")
        file.write(f"Accuracy: {accuracy_score(y_test_cpu, y_pred)}\n")
        file.write(classification_report(y_test_cpu, y_pred))
        file.write(f"\n")
