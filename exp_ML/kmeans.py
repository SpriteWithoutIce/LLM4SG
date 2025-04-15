import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle


def kmeans(trainPath, testPath, outputPath, outputcsv):
    data = pd.read_csv(trainPath)
    data_test = pd.read_csv(testPath)
    data_1 = data["heartbeat_signals"].str.split(",", expand=True)
    data_test_1 = data_test["heartbeat_signals"].str.split(",", expand=True)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    data_2 = np.array(data_1).astype("float32").reshape(-1, 100)
    data_2 = imputer.fit_transform(data_2)
    data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100)
    data_test_2 = imputer.fit_transform(data_test_2)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data_2)

    predictions = kmeans.predict(data_test_2)

    silhouette_avg = silhouette_score(data_2, kmeans.labels_)
    with open(outputPath, 'a') as file:
        file.write(f"K-Means\n")
        file.write(f"Silhouette Score: {silhouette_avg}\n")

    train_predictions = kmeans.predict(data_2)
    test_predictions = kmeans.predict(data_test_2)

    # 限制每个类别的样本数量不超过100
    max_samples_per_cluster = 100
    for cluster in range(kmeans.n_clusters):
        cluster_data = data_2[kmeans.labels_ == cluster]
        if len(cluster_data) > max_samples_per_cluster:
            # 随机抽样，保留100个样本
            cluster_data = shuffle(
                cluster_data[:max_samples_per_cluster], random_state=42)
            # 更新聚类标签
            kmeans.labels_[kmeans.labels_ ==
                           cluster] = np.random.choice(kmeans.labels_)
            kmeans.labels_[np.in1d(kmeans.labels_, cluster)] = cluster

    data['label'] = train_predictions
    data_test['label'] = test_predictions

    combined_data = pd.concat([data, data_test], ignore_index=True)

    final_data = combined_data[['heartbeat_signals', 'label']]

    final_data.to_csv(outputcsv, index=False)
