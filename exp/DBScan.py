import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def dbscan(trainPath, testPath, outputPath, outputcsv):
    data = pd.read_csv(trainPath)
    data_test = pd.read_csv(testPath)
    data_1 = data["heartbeat_signals"].str.split(",", expand=True)
    data_test_1 = data_test["heartbeat_signals"].str.split(",", expand=True)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = StandardScaler()

    data_2 = np.array(data_1).astype("float32").reshape(-1, 100)
    data_2 = imputer.fit_transform(data_2)
    data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100)
    data_test_2 = imputer.fit_transform(data_test_2)

    data_2 = scaler.fit_transform(data_2)
    data_test_2 = scaler.transform(data_test_2)

    # 合并训练数据和测试数据
    combined_data = np.vstack((data_2, data_test_2))

    clustering = DBSCAN(eps=0.5, min_samples=3)
    labels = clustering.fit_predict(combined_data)

    # 分割聚类标签
    train_predictions = labels[:len(data_2)]
    test_predictions = labels[len(data_2):]

    with open(outputPath, 'a') as file:
        file.write(f"DBSCAN\n")

    data['label'] = train_predictions
    data_test['label'] = test_predictions

    combined_data = pd.concat([data, data_test], ignore_index=True)

    final_data = combined_data[['heartbeat_signals', 'label']]

    final_data.to_csv(outputcsv, index=False)
