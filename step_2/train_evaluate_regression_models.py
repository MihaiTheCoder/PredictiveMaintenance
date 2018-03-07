import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def get_top_features(dataset, n_features):
    rul_column = dataset['RUL']
    train_dataset = dataset.iloc[:, 0:48]
    train_correlation = abs(train_dataset.corrwith(rul_column))
    important_features = train_correlation.sort_values(ascending=False)[:n_features]

    dataset_top_features = dataset[list(important_features.index)]
    return dataset_top_features


if __name__ == '__main__':
    df = pd.read_csv("../process_input/train_dataset_1_of_3.csv")

    #  Not useful for regression models
    df.drop(labels=['label1', 'label2'], axis=1, inplace=True)

    #  get top 35 features (35 chosen at random)
    df_top_features = get_top_features(df, 35)

    regressor = GradientBoostingRegressor(n_estimators=100, min_samples_leaf=10, learning_rate=0.2, max_leaf_nodes=20)
    est = regressor.fit(df_top_features.values, df['RUL'].values)

    print(df_top_features)
