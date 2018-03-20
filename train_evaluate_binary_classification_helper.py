import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib


def sample(training_features_df, training_label_values):
    positive_training_labels = np.where(training_label_values == 1)[0]
    negative_training_labels = np.where(training_label_values == 0)[0]
    sample_size = len(positive_training_labels)

    sample_negative_training_labels = np.random.choice(negative_training_labels, sample_size)

    sampled_training_indexes = np.append(positive_training_labels, sample_negative_training_labels)
    sample_training_labels = training_label_values[sampled_training_indexes]

    sample_training_features = training_features_df.filter(sampled_training_indexes, axis='index')
    return sample_training_features, sample_training_labels


def get_binary_classification_predictions(training_features_df, training_label_values, testing_features_df, testing_label_values):
    sample_training_features, sample_training_labels = sample(training_features_df, training_label_values)

    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=20, min_samples_split=10), n_estimators=100,
                                    learning_rate=0.2)
    classifier.fit(training_features_df, training_label_values)
    joblib.dump(classifier, 'binary_classification/AdaBoostDecisionTreeClassifier.pkl')

    ada_classifier_predicted = classifier.predict(testing_features_df)

    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=20, min_samples_split=10), n_estimators=100,
                                    learning_rate=0.2)
    classifier.fit(sample_training_features, sample_training_labels)
    joblib.dump(classifier, 'binary_classification/Sampled_AdaBoostDecisionTreeClassifier.pkl')
    sampled_ada_classifier_predicted = classifier.predict(testing_features_df)

    random_forestClassifier = RandomForestClassifier(n_estimators=8, max_depth=32)
    random_forestClassifier.fit(training_features_df, training_label_values)
    joblib.dump(random_forestClassifier, 'binary_classification/random_forestClassifier.pkl')
    random_forest_predicted = random_forestClassifier.predict(testing_features_df)

    logreg = LogisticRegression(tol=1e-07)
    logreg.fit(training_features_df, training_label_values)
    joblib.dump(logreg, 'binary_classification/LogisticRegression.pkl')
    logreg_predicted = logreg.predict(testing_features_df)

    mlp_classifier = MLPClassifier(learning_rate_init=0.1, solver='sgd', momentum=0, max_iter=100)
    mlp_classifier.fit(training_features_df, training_label_values)
    joblib.dump(mlp_classifier, 'binary_classification/NeuralNetworkMLPClassifier.pkl')
    mlp_predicted = mlp_classifier.predict(testing_features_df)

    predictions_truth = pd.DataFrame({'LogisticRegression_Prediction': pd.Series(logreg_predicted),
                                      'AdaBoostDecisionTreeClassifier_Prediction': pd.Series(ada_classifier_predicted),
                                      'NeuralNetworkMLPClassifier_Prediction': pd.Series(mlp_predicted),
                                      'RandomForest_Prediction': pd.Series(random_forest_predicted),
                                      'Sampled_AdaBoostDecisionTreeClassifier_Prediction': pd.Series(sampled_ada_classifier_predicted),
                                      'label': pd.Series(testing_label_values)})

    return predictions_truth


def get_beamer_table_values(truth, prediction):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(0, len(truth)):
        if truth[i] == prediction[i] and truth[i] == 1:
            true_positive = true_positive + 1
        if truth[i] == prediction[i] and truth[i] == 0:
            true_negative = true_negative + 1
        if truth[i] != prediction[i] and truth[i] == 1:
            false_negative = false_negative + 1
        if truth[i] != prediction[i] and truth[i] == 0:
            false_positive = false_positive + 1

    return true_positive, false_negative, false_positive, true_negative


def evaluate_binary_classification_models(predictions_truth):
    prediction_columns = list(predictions_truth.columns.values)
    prediction_columns.remove('label')
    truth = predictions_truth['label'].values
    list_of_features = list()
    for prediction_column in prediction_columns:
        prediction = predictions_truth[prediction_column].values
        positive_label = 1
        negative_label = 0
        true_positive, false_negative, false_positive, true_negative = get_beamer_table_values(truth, prediction)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        f_score = 2 * (precision * recall) / (precision + recall)
        auc_score = metrics.roc_auc_score(truth, prediction)
        list_of_features.append([true_positive, false_negative, false_positive, true_negative,
                                 positive_label, negative_label, precision, recall, accuracy,
                                 f_score, auc_score])

    return pd.DataFrame(list_of_features, columns=['True Positive', 'False Negative',
                                                   'False Positive', 'True Negative',
                                                   'Positive Label', 'Negative Label',
                                                   'Precision', 'Recall', 'Accuracy',
                                                   'F1 Score', 'AUC'], index=prediction_columns)