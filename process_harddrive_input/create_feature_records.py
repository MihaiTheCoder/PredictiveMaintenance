import pandas as pd
import numpy as np

w0 = 150  # first time window - Cycles
w1 = 300  # second time window - Cycles


def add_time_windows_to_label(dataset):
    dataset['label1'] = np.where(dataset['RUL'] <= w1, 1, 0)  # 1 if RUL <= w1 else 0
    dataset['label2'] = np.where(dataset['RUL'] <= w0, 2, dataset['label1'])  # 2 if RUL <= w0 else label1


def add_RUL_and_time_windos(input_dataset):

    serial_numbers = list(input_dataset['serial_number'].unique())
    merged = []
    for serial_number in serial_numbers:
        records_for_device = input_dataset[input_dataset['serial_number'] == serial_number]
        max_hours = records_for_device['smart_9_raw'].max()
        # RUL = Remaining Useful Life
        records_for_device['RUL'] = max_hours - records_for_device['smart_9_raw']
        add_time_windows_to_label(records_for_device)
        merged.append(records_for_device)

    pd.concat(merged)
    return merged

if __name__ == '__main__':
    train_df = pd.read_csv('training_harddrive.csv')

    rul_added = add_RUL_and_time_windos(train_df)

    print(rul_added)


