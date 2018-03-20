import pandas as pd
import sys
sys.path.append('../')
import train_evaluate_binary_classification_helper


def get_top_features(dataset, n_features):
    label_column = dataset['label1']
    train_dataset = dataset.iloc[:, 0:48]
    train_correlation = abs(train_dataset.corrwith(label_column))
    important_features = train_correlation.sort_values(ascending=False)[:n_features]

    return list(important_features.index)


if __name__ == '__main__':
    df_train = pd.read_csv("../process_input/train_dataset_1_of_3.csv")
    test_df = pd.read_csv("../process_input/test_dataset_1_of_3.csv")

    #  Not useful for regression models
    df_train.drop(labels=['RUL', 'label2'], axis=1, inplace=True)

    top_features = get_top_features(df_train, 35)
    #  get top 35 features (35 chosen at random)
    df_train_top_features = df_train[top_features]

    # features to use for prediction
    test_df_features = test_df[top_features]

    predictions_truth = train_evaluate_binary_classification_helper.get_binary_classification_predictions(
        df_train_top_features, df_train['label1'].values, test_df_features, test_df['label1'].values)
    evaluation = train_evaluate_binary_classification_helper.evaluate_binary_classification_models(predictions_truth)
    evaluation.to_csv('binary_classification/comparison.csv', index_label='Algorithm')
    print(evaluation)
