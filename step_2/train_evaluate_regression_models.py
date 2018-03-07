import pandas as pd

def get_top_features(dataset, n_features):
    rul_column = dataset['RUL']
    train_dataset = dataset.iloc[:, 0:48]
    train_correlation = abs(train_dataset.corrwith(rul_column))
    important_features = train_correlation.sort_values(ascending=False)[:n_features]

    dataset_top_features = dataset[list(important_features.index)]
    dataset_top_features['RUL'] = dataset['RUL'].values
    return dataset_top_features

if __name__ == '__main__':
    df = pd.read_csv("../process_input/train_dataset_1_of_3.csv")

    #  Not useful for regression models
    df.drop(labels=['label1', 'label2'], axis=1, inplace=True)

    #  get top 35 features (35 random..)
    df_top_features = list(get_top_features(df, 35).index)

    print(df_top_features)
