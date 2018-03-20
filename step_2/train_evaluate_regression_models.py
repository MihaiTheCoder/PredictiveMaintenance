import pandas as pd
import sys
sys.path.append('../')
import train_evaluate_regression_models_helper


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

    top_features_text = "\n".join(top_features)

    with open('important_features.txt', 'w') as important_features_file:
        print(top_features_text, file=important_features_file)

    # features to use for prediction
    test_df_features = test_df[top_features]

    predictions_truth = train_evaluate_regression_models_helper.get_regression_predictions(df_train_top_features,
                                                                                           df_train['RUL'].values,
                                                                                           test_df_features,
                                                                                           test_df['RUL'].values)
    evaluation = train_evaluate_regression_models_helper.evaluate_regression_models(predictions_truth)
    evaluation.to_csv('regression/comparison.csv', index_label='Algorithm')

    print(evaluation)
