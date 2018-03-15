import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import create_feature_records_helper

# Multi-class classification: Predict if an asset will fail in different time windows:
# E.g., fails in window [1, w0] days; fails in the window [w0+1,w1] days; not fail within w1 days

w0 = 15  # first time window - Cycles
w1 = 30  # second time window - Cycles

# window size (window_size>=2),  most recent sensor values
window_size = 5

labels = ['RUL', 'label1', 'label2']
pre_sensor_columns = ['cycle', 'setting1', 'setting2']

n_pre_sensor_columns = 5  # id, cycle, setting1,setting2,setting3
n_train_after_sensor_columns = 3  # RUL, label1, label2
n_test_after_sensor_columns = 0
initial_column_names = ["id", "cycle", "setting1", "setting2", "setting3", "s1", "s2", "s3", "s4", "s5", "s6",
                        "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
                        "s20", "s21"]


def create_dataset_from_input(tsv_file):
    tsv = pd.read_csv(filepath_or_buffer=tsv_file, delim_whitespace=True, header=None,
                      names=initial_column_names)
    return tsv


def add_time_windows_to_label(dataset):
    dataset['label1'] = np.where(dataset['RUL'] <= w1, 1, 0)  # 1 if RUL <= w1 else 0
    dataset['label2'] = np.where(dataset['RUL'] <= w0, 2, dataset['label1'])  # 2 if RUL <= w0 else label1


# This function generates the labels for the training data.
# RUL (Remaining Useful life) for regression models, label1 for binary classification models,
# label2 for multi-class classification models
def add_RUL_and_time_windows(input_dataset):
    count_df = input_dataset.groupby(['id']).size().reset_index(name='count')

    merged = pd.merge(input_dataset, count_df, on="id")
    # RUL = Remaining Useful Life (RUL)
    merged['RUL'] = merged['count'] - merged['cycle']
    add_time_windows_to_label(merged)
    merged = merged.drop('count', axis=1)

    return merged


def get_sensor_data(dataset, n_after_sensor_columns):
    return create_feature_records_helper.get_sensor_data(dataset, n_pre_sensor_columns, n_after_sensor_columns)



def get_aggregates(dataset, id, rolling_average_columns, rolling_std_columns, n_after_sensor_columns):
    return create_feature_records_helper.get_aggregates(dataset, id, rolling_average_columns, rolling_std_columns,
                                                        n_pre_sensor_columns, n_after_sensor_columns)


# This function performs feature engineering to add more features in the training data. Three additional features
# are created for each of the 21 sensors
# Generated features:
# a1-a21 - moving average of the corresponding sensor measure within the predefined window size w
# sd1-sd21 - moving standard deviation  of the corresponding sensor measure within the predefined window size w
def add_moving_average_moving_std(dataset, n_after_sensor_columns):
    return create_feature_records_helper.add_moving_average_moving_std(dataset, n_pre_sensor_columns,
                                                                       n_after_sensor_columns)


def normalize_relative_to_dataset(dataset, refence_dataset):
    normalized_dataset = (dataset - refence_dataset.min()) / (refence_dataset.max()-refence_dataset.min())
    return normalized_dataset


def get_latest_for_each_id(dataset):
    latest_records = dataset.groupby('id').tail(1)
    return latest_records


def read_file_line_by_line_into_list(file_path):
    with open(file_path) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    return content


if __name__ == '__main__':
    train_tsv = create_dataset_from_input("..\PM_train.txt")
    train_dataset = add_RUL_and_time_windows(train_tsv)
    train_dataset = add_moving_average_moving_std(train_dataset, n_train_after_sensor_columns)

    train_dataset = train_dataset.drop('id', axis=1)
    train_dataset.to_csv('train_dataset_with_extra_features.csv')
    normalized_train_dataset = create_feature_records_helper.normalize_all_except(train_dataset, labels)
    normalized_train_dataset = create_feature_records_helper.set_labels_as_latest_columns(normalized_train_dataset,
                                                                                          labels)
    normalized_train_dataset.dropna(axis=1, inplace=True)
    normalized_train_dataset.to_csv("train_dataset_1_of_3.csv", index=False, header=True)

    test_tsv = create_dataset_from_input("..\PM_test.txt")
    test_dataset = add_moving_average_moving_std(test_tsv, n_test_after_sensor_columns)
    test_dataset = get_latest_for_each_id(test_dataset)

    test_dataset = test_dataset.drop('id', axis=1)
    normalized_test_dataset = normalize_relative_to_dataset(test_dataset, train_dataset)
    normalized_test_dataset.dropna(axis=1, inplace=True)
    normalized_test_dataset = create_feature_records_helper.set_pre_sensors_first(normalized_test_dataset, pre_sensor_columns)

    scores = pd.DataFrame({'RUL': pd.Series(read_file_line_by_line_into_list('..\PM_truth.txt'), dtype='int32')})
    add_time_windows_to_label(scores)
    normalized_test_dataset = normalized_test_dataset.reset_index()
    normalized_test_dataset = normalized_test_dataset.merge(scores, left_index=True, right_index=True)
    normalized_test_dataset.to_csv("test_dataset_1_of_3.csv", index=False, header=True)
