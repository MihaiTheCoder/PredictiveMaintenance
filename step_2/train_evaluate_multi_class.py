import pandas as pd
import sys
sys.path.append('../')
import train_evaluate_multiclass_helper


def get_top_features(dataset, n_features):
    label_column = dataset['label2']
    train_dataset = dataset.iloc[:, 0:48]
    train_correlation = abs(train_dataset.corrwith(label_column))
    important_features = train_correlation.sort_values(ascending=False)[:n_features]

    return list(important_features.index)


if __name__ == '__main__':
    df_train = pd.read_csv("../process_input/train_dataset_1_of_3.csv")
    test_df = pd.read_csv("../process_input/test_dataset_1_of_3.csv")

    #  Not useful for regression models
    df_train.drop(labels=['RUL', 'label1'], axis=1, inplace=True)

    top_features = get_top_features(df_train, 35)
    #  get top 35 features (35 chosen at random)
    df_train_top_features = df_train[top_features]

    # features to use for prediction
    test_df_features = test_df[top_features]

    predictions_truth = train_evaluate_multiclass_helper.get_multiclass_classification_predictions(
        df_train_top_features, df_train['label2'].values, test_df_features, test_df['label2'].values)
    evaluation = train_evaluate_multiclass_helper.evaluate_multiclass_classification_models(predictions_truth)
    evaluation.to_csv('multiclass/comparison.csv', index_label='Algorithm')
    print(evaluation)
