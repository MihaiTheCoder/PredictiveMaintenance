import pandas as pd
import numpy as np


def get_sensor_data(dataset, n_pre_sensor_columns, n_after_sensor_columns):
    sensor_data_columns = dataset.columns[n_pre_sensor_columns: len(dataset.columns) - n_after_sensor_columns]
    sensor_data = dataset[sensor_data_columns]
    return sensor_data


def get_aggregates(dataset, id, rolling_average_columns, rolling_std_columns, n_pre_sensor_columns,
                   n_after_sensor_columns):
    sub_data = get_sensor_data(dataset[dataset['id'] == id], n_pre_sensor_columns, n_after_sensor_columns)
    rolling_average = sub_data.rolling(5, min_periods=1).mean()
    rolling_average.columns = rolling_average_columns
    rolling_average['id'] = id
    rolling_std = sub_data.rolling(5, min_periods=1).std().fillna(0)
    rolling_std.columns = rolling_std_columns
    rolling_std['id'] = id

    aggregates = rolling_average.merge(rolling_std, left_index=True, right_index=True)
    aggregates = recreate_id_column(aggregates)
    return aggregates


def recreate_id_column(dataset):
    dataset = dataset.drop('id_x', axis=1)
    dataset = dataset.rename(columns={'id_y': 'id'})
    return dataset


def add_moving_average_moving_std(dataset, n_pre_sensor_columns, n_after_sensor_columns):
    n_id = len(dataset['id'].unique())

    number_of_sensors = len(dataset.columns) - n_pre_sensor_columns - n_after_sensor_columns
    rolling_average_columns = ["a{}".format(i) for i in range(1, number_of_sensors + 1)]
    rolling_std_columns = ['sd{}'.format(i) for i in range(1, number_of_sensors + 1)]
    aggregates = [get_aggregates(dataset, i, rolling_average_columns, rolling_std_columns, n_pre_sensor_columns,
                                 n_after_sensor_columns) for i in range(1, n_id + 1)]

    aggregates = pd.concat(aggregates)
    merged_dataset = dataset.merge(aggregates, left_index=True, right_index=True)
    merged_dataset = recreate_id_column(merged_dataset)

    return merged_dataset


def set_labels_as_latest_columns(dataset, labels):
    columns = list(dataset.columns.values)
    for label in labels:
        columns.remove(label)
        columns.append(label)
    return dataset[columns]


def normalize_relative_to_dataset(dataset, refence_dataset):
    normalized_dataset = (dataset - refence_dataset.min()) / (refence_dataset.max()-refence_dataset.min())
    return normalized_dataset


def normalize_all_except(dataset, excluded_columns):
    normalized_dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    for excluded_column in excluded_columns:
        normalized_dataset[excluded_column] = dataset[excluded_column]
    return normalized_dataset


def set_pre_sensors_first(dataset, pre_sensor_columns):
    columns = list(dataset.columns.values)
    for pre_sensor_column in reversed(pre_sensor_columns):
        columns.remove(pre_sensor_column)
        columns.insert(0, pre_sensor_column)
    return dataset[columns]


def set_post_sensor_last(dataset, post_sensor_columns):
    columns = list(dataset.columns.values)
    for post_sensor_column in post_sensor_columns:
        columns.remove(post_sensor_column)
        columns.insert(len(columns), post_sensor_column)
    return dataset[columns]

