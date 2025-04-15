import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(file_path):
    return pd.read_csv(file_path)


def split_dataset(df, test_size=0.2, random_state=None):
    df_shuffled = df.sample(
        frac=1, random_state=random_state).reset_index(drop=True)
    return train_test_split(df_shuffled, test_size=test_size, random_state=random_state, stratify=df_shuffled['label'])


def write_csv(df, file_name):
    df.to_csv(file_name, index=False)


def main():
    df = read_csv('./12042.csv')  # 替换为你的CSV文件路径

    train_df, test_df = split_dataset(df, test_size=0.2, random_state=42)

    new_df = test_df.iloc[:, :2]

    train_df.columns = ['id', 'heartbeat_signals', 'label']
    test_df.columns = ['id', 'heartbeat_signals', 'answer']
    new_df.columns = ['id', 'heartbeat_signals']

    write_csv(train_df, 'train_dataset_1204_3.csv')
    write_csv(test_df, 'test_dataset_1204_3.csv')
    write_csv(new_df, 'test_label_1204_3.csv')
    print("数据集划分完成，train和test文件已生成。")


# 调用主函数
if __name__ == '__main__':
    main()
