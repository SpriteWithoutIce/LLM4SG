import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def hierarchical_clustering(trainPath, testPath, outputPath, outputcsv):
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

    clustering = AgglomerativeClustering(n_clusters=3)
    labels = clustering.fit_predict(np.vstack((data_2, data_test_2)))

    silhouette_avg = silhouette_score(data_2, labels[:len(data_2)])
    with open(outputPath, 'a') as file:
        file.write(f"Hierarchical Clustering\n")
        file.write(f"Silhouette Score: {silhouette_avg}\n")

    train_predictions = labels[:len(data_2)]
    test_predictions = labels[len(data_2):]

    data['label'] = train_predictions
    data_test['label'] = test_predictions

    combined_data = pd.concat([data, data_test], ignore_index=True)

    final_data = combined_data[['heartbeat_signals', 'label']]

    final_data.to_csv(outputcsv, index=False)
