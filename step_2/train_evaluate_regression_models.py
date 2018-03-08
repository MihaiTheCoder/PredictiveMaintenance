import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def get_top_features(dataset, n_features):
    rul_column = dataset['RUL']
    train_dataset = dataset.iloc[:, 0:48]
    train_correlation = abs(train_dataset.corrwith(rul_column))
    important_features = train_correlation.sort_values(ascending=False)[:n_features]

    return list(important_features.index)


if __name__ == '__main__':
    df_train = pd.read_csv("../process_input/train_dataset_1_of_3.csv")
    test_df = pd.read_csv("../process_input/test_dataset_1_of_3.csv")

    #  Not useful for regression models
    df_train.drop(labels=['label1', 'label2'], axis=1, inplace=True)

    top_features = get_top_features(df_train, 35)
    #  get top 35 features (35 chosen at random)
    df_train_top_features = df_train[top_features]

    regressor = GradientBoostingRegressor(n_estimators=100, min_samples_leaf=10, learning_rate=0.2, max_leaf_nodes=20)
    est = regressor.fit(df_train_top_features.values, df_train['RUL'].values)

    # features to use for prediction
    test_df_features = test_df[top_features]

    predictions = est.predict(test_df_features.values)

    predictions_truth = pd.DataFrame({'Prediction': pd.Series(predictions), 'RUL': pd.Series(test_df['RUL'].values)})
    print(predictions_truth)


    #print(predictions_truth)
