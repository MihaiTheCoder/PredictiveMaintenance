import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def get_top_features(dataset, n_features):
    label_column = dataset['label1']
    train_dataset = dataset.iloc[:, 0:48]
    train_correlation = abs(train_dataset.corrwith(label_column))
    important_features = train_correlation.sort_values(ascending=False)[:n_features]

    return list(important_features.index)


def get_binary_classification_predictions(training_features_df, training_label_values, testing_features_df, testing_label_values):
    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=20, min_samples_split=10), n_estimators=100,
                                    learning_rate=0.2)
    classifier.fit(training_features_df, training_label_values)

    ada_classifier_predicted = classifier.predict(testing_features_df)

    random_forestClassifier = RandomForestClassifier(n_estimators=8, max_depth=32)
    random_forestClassifier.fit(training_features_df, training_label_values)
    random_forest_predicted = random_forestClassifier.predict(testing_features_df)


    logreg = LogisticRegression(tol=1e-7)
    logreg.fit(training_features_df, training_label_values)
    logreg_predicted = logreg.predict(testing_features_df)

    predictions_truth = pd.DataFrame({'LogisticRegression_Prediction': pd.Series(logreg_predicted),
                                      'AdaBoostDecisionTreeClassifier_Prediction': pd.Series(ada_classifier_predicted),
                                      'RandomForest_Prediction': pd.Series(random_forest_predicted),
                                      'label': pd.Series(testing_label_values)})

    return predictions_truth


def evaluate_bianry_classification_models(predictions_truth):
    return predictions_truth


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

    predictions_truth = get_binary_classification_predictions(df_train_top_features, df_train['label1'].values,
                                                              test_df_features, test_df['label1'].values)
    evaluation = evaluate_bianry_classification_models(predictions_truth)
    print(evaluation)
