import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.externals import joblib
from mord import LogisticIT, LAD, LogisticSE
from math import sqrt


def get_top_features(dataset, n_features):
    label_column = dataset['label2']
    train_dataset = dataset.iloc[:, 0:48]
    train_correlation = abs(train_dataset.corrwith(label_column))
    important_features = train_correlation.sort_values(ascending=False)[:n_features]

    return list(important_features.index)


def get_multiclass_classification_predictions(training_features_df, training_label_values, testing_features_df, testing_label_values):

    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=20, min_samples_split=10), n_estimators=100,
                                    learning_rate=0.2)
    classifier.fit(training_features_df, training_label_values)
    joblib.dump(classifier, 'multiclass/AdaBoostDecisionTreeClassifier.pkl')
    ada_classifier_predicted = classifier.predict(testing_features_df)

    random_forestClassifier = RandomForestClassifier(n_estimators=8, max_depth=32)
    random_forestClassifier.fit(training_features_df, training_label_values)
    joblib.dump(random_forestClassifier, 'multiclass/random_forestClassifier.pkl')
    random_forest_predicted = random_forestClassifier.predict(testing_features_df)


    logreg = LogisticRegression(tol=1e-07)
    logreg.fit(training_features_df, training_label_values)
    joblib.dump(logreg, 'multiclass/LogisticRegression.pkl')
    logreg_predicted = logreg.predict(testing_features_df)


    mlp_classifier = MLPClassifier(learning_rate_init=0.1, solver='sgd', momentum=0, max_iter=100)
    mlp_classifier.fit(training_features_df, training_label_values)
    joblib.dump(mlp_classifier, 'multiclass/NeuralNetworkMLPClassifier.pkl')
    mlp_predicted = mlp_classifier.predict(testing_features_df)


    nn_ordinal_regression = LogisticIT()
    nn_ordinal_regression.fit(training_features_df, training_label_values)
    joblib.dump(nn_ordinal_regression, 'multiclass/nn_ordinal_regression.pkl')
    nn_ordinal_regression_predicted = nn_ordinal_regression.predict(testing_features_df)

    lr_ordinal_regression = LogisticSE()
    lr_ordinal_regression.fit(training_features_df, training_label_values)
    joblib.dump(lr_ordinal_regression, 'multiclass/lr_ordinal_regression.pkl')
    lr_ordinal_regression_predicted = lr_ordinal_regression.predict(testing_features_df)


    predictions_truth = pd.DataFrame({'LogisticRegression_Prediction': pd.Series(logreg_predicted),
                                      'AdaBoostDecisionTreeClassifier_Prediction': pd.Series(ada_classifier_predicted),
                                      'NeuralNetworkMLPClassifier_Prediction': pd.Series(mlp_predicted),
                                      'RandomForest_Prediction': pd.Series(random_forest_predicted),
                                      'Ordinal Regression NN': pd.Series(nn_ordinal_regression_predicted),
                                      'Ordinal Regression LR': pd.Series(lr_ordinal_regression_predicted),
                                      'label': pd.Series(testing_label_values)})

    return predictions_truth


def get_confusion_matrix_multiclass(truth, prediction):
    return metrics.confusion_matrix(y_true=truth, y_pred=prediction)


def evaluate_multiclass_classification_models(predictions_truth):
    prediction_columns = list(predictions_truth.columns.values)
    prediction_columns.remove('label')
    truth = predictions_truth['label'].values
    list_of_features = list()
    for prediction_column in prediction_columns:
        prediction = predictions_truth[prediction_column].values
        mat = get_confusion_matrix_multiclass(truth, prediction)

        #accuracy
        overall_accuracy = metrics.accuracy_score(y_true=truth, y_pred=prediction)
        sum_avg = 0
        for k in range(0, len(mat)):
            S = mat[k][k] / mat.sum(axis=1)[k]
            sum_avg = sum_avg + S
        avg_accuracy = sum_avg / len(mat)

        #precision
        micro_precision = metrics.precision_score(y_true=truth, y_pred=prediction, average='micro')
        macro_precision = metrics.precision_score(y_true=truth, y_pred=prediction, average='macro')

        #recall
        micro_recall = metrics.recall_score(y_true=truth, y_pred=prediction, average='micro')
        macro_recall = metrics.recall_score(y_true=truth, y_pred=prediction, average='macro')

        mean_zero_one_error = metrics.zero_one_loss(truth, prediction)
        mean_absolute_error = metrics.mean_absolute_error(truth, prediction)
        root_mean_squared_error = sqrt(metrics.mean_squared_error(truth, prediction))


        list_of_features.append([overall_accuracy, avg_accuracy, micro_precision, macro_precision,
                                 micro_recall, macro_recall, mean_zero_one_error,mean_absolute_error,root_mean_squared_error])

    return pd.DataFrame(list_of_features, columns=['Overall Accuracy', 'Averaged Accuracy',
                                                   'Micro-averaged Precision', 'Macro-averaged Precision',
                                                   'Micro-averaged Recall', 'Macro-averaged Recall',
                                                   'Mean Zero One Error', 'Mean Absolute Error', 'Root Mean Squared Error'],
                                                    index=prediction_columns)


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

    predictions_truth = get_multiclass_classification_predictions(df_train_top_features, df_train['label2'].values,
                                                                  test_df_features, test_df['label2'].values)
    evaluation = evaluate_multiclass_classification_models(predictions_truth)
    evaluation.to_csv('multiclass/comparison.csv', index_label='Algorithm')
    print(evaluation)
