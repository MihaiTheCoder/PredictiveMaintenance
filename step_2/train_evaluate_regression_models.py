import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


def get_top_features(dataset, n_features):
    rul_column = dataset['RUL']
    train_dataset = dataset.iloc[:, 0:48]
    train_correlation = abs(train_dataset.corrwith(rul_column))
    important_features = train_correlation.sort_values(ascending=False)[:n_features]

    return list(important_features.index)


def get_regression_predictions(training_features_df, training_rul_values, testing_features_df, testing_rul_values):
    gradient_boosting_regressor = GradientBoostingRegressor(n_estimators=100, min_samples_leaf=10, learning_rate=0.2, max_leaf_nodes=20)
    gradient_boosting_esimator = gradient_boosting_regressor.fit(training_features_df.values, training_rul_values)
    gradient_boosting_predictions = gradient_boosting_esimator.predict(testing_features_df.values)

    decision_forest_regressor = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_leaf_nodes=20)
    decision_forest_estimator = decision_forest_regressor.fit(df_train_top_features.values, df_train['RUL'].values)
    decision_forest_predictions = decision_forest_estimator.predict(testing_features_df.values)

    predictions_truth = pd.DataFrame({'GradientBoostingRegressor_Prediction': pd.Series(gradient_boosting_predictions),
                                      'DecisionForestRegressor_Prediction': pd.Series(decision_forest_predictions),
                                      'RUL': pd.Series(testing_rul_values)})

    #  RUL GradientBoostingRegressor_Prediction DecisionForestRegression_Prediction
    return predictions_truth


if __name__ == '__main__':
    df_train = pd.read_csv("../process_input/train_dataset_1_of_3.csv")
    test_df = pd.read_csv("../process_input/test_dataset_1_of_3.csv")

    #  Not useful for regression models
    df_train.drop(labels=['label1', 'label2'], axis=1, inplace=True)

    top_features = get_top_features(df_train, 35)
    #  get top 35 features (35 chosen at random)
    df_train_top_features = df_train[top_features]

    # features to use for prediction
    test_df_features = test_df[top_features]

    predictions_truth = get_regression_predictions(df_train_top_features, df_train['RUL'].values, test_df_features, test_df['RUL'].values)

    print(predictions_truth)
