import pandas as pd
import numpy as np

# Multi-class classification: Predict if an asset will fail in different time windows:
# E.g., fails in window [1, w0] days; fails in the window [w0+1,w1] days; not fail within w1 days

w0 = 15  # first time window - Cycles
w1 = 30  # second time window - Cycles

# window size (window_size>=2),  most recent sensor values
window_size = 5

labels = ['RUL', 'label1', 'label2']

n_pre_sensor_columns = 5  # id, cycle, setting1,setting2,setting3
n_train_after_sensor_columns = 3  # RUL, label1, label2
n_test_after_sensor_columns = 0


def create_dataset_from_input(tsv_file):
    tsv = pd.read_csv(tsv_file, delim_whitespace=True, header=None,
                      names=["id", "cycle", "setting1", "setting2", "setting3", "s1", "s2", "s3", "s4", "s5", "s6",
                             "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
                             "s20", "s21"])
    return tsv


# This function generates the labels for the training data.
# RUL (Remaining Useful life) for regression models, label1 for binary classification models,
# label2 for multi-class classification models
def add_RUL_and_time_windows(input_dataset):
    count_df = input_dataset.groupby(['id']).size().reset_index(name='count')

    merged = pd.merge(input_dataset, count_df, on="id")
    # RUL = Remaining Useful Life (RUL)
    merged['RUL'] = merged['count'] - merged['cycle']
    merged['label1'] = np.where(merged['RUL'] <= w1, 1, 0)  # 1 if RUL <= w1 else 0
    merged['label2'] = np.where(merged['RUL'] <= w0, 2, merged['label1'])  # 2 if RUL <= w0 else label1
    merged = merged.drop('count', axis=1)

    return merged


def get_sensor_data(dataset, n_after_sensor_columns):
    sensor_data_columns = dataset.columns[n_pre_sensor_columns: len(dataset.columns) - n_after_sensor_columns]
    sensor_data = dataset[sensor_data_columns]
    return sensor_data


def recreate_id_column(dataset):
    dataset = dataset.drop('id_x', axis=1)
    dataset = dataset.rename(columns={'id_y': 'id'})
    return dataset


def get_aggregates(dataset, id, rolling_average_columns, rolling_std_columns, n_after_sensor_columns):
    sub_data = get_sensor_data(dataset[dataset['id'] == id], n_after_sensor_columns)
    rolling_average = sub_data.rolling(5, min_periods=1).mean()
    rolling_average.columns = rolling_average_columns
    rolling_average['id'] = id
    rolling_std = sub_data.rolling(5, min_periods=1).std().fillna(0)
    rolling_std.columns = rolling_std_columns
    rolling_std['id'] = id

    aggregates = rolling_average.merge(rolling_std, left_index=True, right_index=True)
    aggregates = recreate_id_column(aggregates)
    return aggregates


# This function performs feature engineering to add more features in the training data. Three additional features
# are created for each of the 21 sensors
# Generated features:
# a1-a21 - moving average of the corresponding sensor measure within the predefined window size w
# sd1-sd21 - moving standard deviation  of the corresponding sensor measure within the predefined window size w
def add_moving_average_moving_std(dataset, n_after_sensor_columns):
    n_id = len(dataset['id'].unique())

    number_of_sensors = len(dataset.columns) - n_pre_sensor_columns - n_after_sensor_columns
    rolling_average_columns = ["a{}".format(i) for i in range(1, number_of_sensors + 1)]
    rolling_std_columns = ['sd{}'.format(i) for i in range(1, number_of_sensors + 1)]
    aggregates = [get_aggregates(dataset, i, rolling_average_columns, rolling_std_columns, n_after_sensor_columns) for i
                  in range(1, n_id + 1)]

    aggregates = pd.concat(aggregates)
    merged_dataset = dataset.merge(aggregates, left_index=True, right_index=True)
    merged_dataset = recreate_id_column(merged_dataset)

    return merged_dataset


def normalize_all_except(dataset, excluded_columns):
    normalized_dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    for excluded_column in excluded_columns:
        normalized_dataset[excluded_column] = dataset[excluded_column]
    return normalized_dataset


def set_labels_as_latest_columns(dataset, labels):
    columns = list(dataset.columns.values)
    for label in labels:
        columns.remove(label)
        columns.append(label)
    return dataset[columns]


if __name__ == '__main__':
    train_tsv = create_dataset_from_input("..\PM_train.txt")
    train_dataset = add_RUL_and_time_windows(train_tsv)
    train_dataset = add_moving_average_moving_std(train_dataset, n_train_after_sensor_columns)
    train_dataset = train_dataset.drop('id', axis=1)
    normalized_train_dataset = normalize_all_except(train_dataset, labels)
    normalized_train_dataset = set_labels_as_latest_columns(normalized_train_dataset, labels)
    normalized_train_dataset.dropna(axis=1, inplace=True)
    normalized_train_dataset.to_csv("train_dataset_1_of_3.csv", index=False, header=True)