import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import create_feature_records_helper


w0 = 150  # first time window - Cycles
w1 = 300  # second time window - Cycles
columns_to_remove_before_prediction = ['date', 'serial_number', 'model', 'smart_9_raw', 'failure']
labels = ['RUL', 'label1', 'label2']

def add_time_windows_to_label(dataset):
    dataset['label1'] = np.where(dataset['RUL'] <= w1, 1, 0)  # 1 if RUL <= w1 else 0
    dataset['label2'] = np.where(dataset['RUL'] <= w0, 2, dataset['label1'])  # 2 if RUL <= w0 else label1


def add_RUL_and_time_windos(input_dataset):
    serial_numbers = list(input_dataset['serial_number'].unique())
    merged = []
    for serial_number in serial_numbers:
        records_for_device = input_dataset[input_dataset['serial_number'] == serial_number].copy()

        records_for_device.insert(0, 'id', serial_numbers.index(serial_number) + 1)

        max_hours = records_for_device['smart_9_raw'].max()
        # RUL = Remaining Useful Life
        records_for_device['RUL'] = max_hours - records_for_device['smart_9_raw']
        add_time_windows_to_label(records_for_device)
        merged.append(records_for_device)

    merged = pd.concat(merged)
    merged = merged.drop(columns_to_remove_before_prediction, axis=1)
    return merged


if __name__ == '__main__':
    train_df = pd.read_csv('training_harddrive.csv')

    df_extra_features = add_RUL_and_time_windos(train_df)

    df_extra_features = create_feature_records_helper.add_moving_average_moving_std(df_extra_features, 2, 3)

    train_dataset = df_extra_features.drop('id', axis=1)

    df_extra_features.to_csv('train_dataset_with_extra_features.csv', index=False)

    normalized_train_dataset = create_feature_records_helper.normalize_all_except(df_extra_features, labels)
    normalized_train_dataset = create_feature_records_helper.set_labels_as_latest_columns(normalized_train_dataset,
                                                                                          labels)

    normalized_train_dataset.dropna(axis=1, inplace=True)
    normalized_train_dataset.to_csv("train_dataset_1_of_3.csv", index=False, header=True)

    test_df = pd.read_csv('testing_harddrive.csv')
    test_df = add_RUL_and_time_windos(test_df)
    test_df = create_feature_records_helper.add_moving_average_moving_std(test_df, n_pre_sensor_columns=2,
                                                                          n_after_sensor_columns=3)
    labels_df = pd.DataFrame(columns=labels)
    labels_df[labels] = test_df[labels]
    test_df = test_df.drop(['id'] + labels, axis=1)
    normalized_test_dataset = create_feature_records_helper.normalize_relative_to_dataset(test_df, train_dataset)
    normalized_test_dataset.dropna(axis=1, inplace=True)

    normalized_test_dataset = normalized_test_dataset.merge(labels_df,left_index=True, right_index=True)


    normalized_test_dataset.to_csv('test_dataset_1of_3.csv', index=False)



