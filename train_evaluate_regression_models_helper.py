import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.discrete.discrete_model import Poisson


def get_regression_predictions(training_features_df, training_rul_values, testing_features_df, testing_rul_values):
    gradient_boosting_regressor = GradientBoostingRegressor(n_estimators=100, min_samples_leaf=10, learning_rate=0.2,
                                                            max_leaf_nodes=20)
    gradient_boosting_estimator = gradient_boosting_regressor.fit(training_features_df.values, training_rul_values)
    joblib.dump(gradient_boosting_estimator, 'regression/gradient_boosting_estimator.pkl')

    gradient_boosting_predictions = gradient_boosting_estimator.predict(testing_features_df.values)

    decision_forest_regressor = RandomForestRegressor(n_estimators=8, max_depth=32, min_samples_split=128,
                                                      min_samples_leaf=1)
    decision_forest_estimator = decision_forest_regressor.fit(training_features_df.values, training_rul_values)
    joblib.dump(decision_forest_estimator, 'regression/decision_forest_estimator.pkl')
    decision_forest_predictions = decision_forest_estimator.predict(testing_features_df.values)

    neural_network_regressor = MLPRegressor(hidden_layer_sizes=(100,), learning_rate='constant',
                                            learning_rate_init=0.005, max_iter=100)
    neural_network_estimator = neural_network_regressor.fit(training_features_df.values, training_rul_values)
    joblib.dump(neural_network_estimator, 'regression/neural_network_estimator.pkl')
    neural_network_predictions = neural_network_estimator.predict(testing_features_df.values)

    poisson_regressor = Poisson(training_rul_values, training_features_df.values)
    poisson_estimator = poisson_regressor.fit(method="lbfgs", maxiter=20, full_output=False, disp=False)
    joblib.dump(poisson_estimator, 'regression/poisson_estimator.pkl')
    poisson_predictions = poisson_estimator.predict(testing_features_df.values)

    predictions_truth = pd.DataFrame({'GradientBoostingRegressor_Prediction': pd.Series(gradient_boosting_predictions),
                                      'DecisionForestRegressor_Prediction': pd.Series(decision_forest_predictions),
                                      'MLPerceptronRegressor_Prediction': pd.Series(neural_network_predictions),
                                      'PoissonRegressor_Prediction': pd.Series(poisson_predictions),
                                      'RUL': pd.Series(testing_rul_values)})

    #  RUL GradientBoostingRegressor_Prediction DecisionForestRegression_Prediction
    return predictions_truth


def get_root_mean_squared_error(y_true, y_predicted):
    return np.sqrt(((y_true-y_predicted) ** 2).mean())  # sqrt((1/n)*sum((truth-predicted)**2))


def get_relative_absolute_error(y_true, y_predicted):  # mean(|truth-predicted|)
    absolute_error = np.abs(y_true-y_predicted)
    absolute_error_mean = np.abs(y_true-y_true.mean())
    return sum(absolute_error)/sum(absolute_error_mean)


def get_relative_squared_error(y_true, y_predicted):
    squared_error = (y_true-y_predicted) ** 2
    squared_error_mean = (y_true-y_true.mean()) ** 2

    return squared_error.sum()/squared_error_mean.sum()


def evaluate_regression_models(predictions_truth):
    prediction_columns = list(predictions_truth.columns.values)
    prediction_columns.remove('RUL')
    truth = predictions_truth['RUL'].values
    list_of_features = list()
    for prediction_column in prediction_columns:
        prediction = predictions_truth[prediction_column].values
        log_loss = 0#metrics.log_loss(truth, prediction)
        mean_absolute_error = metrics.mean_absolute_error(truth, prediction)
        root_mean_squared_error = get_root_mean_squared_error(truth, prediction)
        relative_absolute_error = get_relative_absolute_error(truth, prediction)
        relative_squared_error = get_relative_squared_error(truth, prediction)
        r2_score = metrics.r2_score(truth, prediction)
        list_of_features.append([log_loss, mean_absolute_error, root_mean_squared_error, relative_absolute_error,
                                 relative_squared_error, r2_score])

    return pd.DataFrame(list_of_features, columns=['Negative Log Likelihood', 'Mean Absolute Error',
                                                   'Root Mean Squared Error', 'Relative Absolute Error',
                                                   'Relative Squared Error', 'Coefficient of Determination'],
                        index=prediction_columns)