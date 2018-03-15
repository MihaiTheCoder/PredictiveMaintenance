import pandas as pd
import random

def get_last_100(df, serial_number):
    return df[df['serial_number'] == serial_number].sort_values(by=['date']).tail(100)

brands = ['HGST', 'Hitachi', 'ST', 'TOSHIBA', 'WDC']
if __name__ == '__main__':
    df = pd.read_csv('..\harddrive.csv\harddrive.csv')
    df = df[df.columns.drop(list(df.filter(regex='normalized')))]

    models = df['model'].unique()

    with open('models.txt', 'w') as file_handler:
        for item in models:
            file_handler.write("{}\n".format(item))


    failed_serial_numbers = list(df[df['failure'] == 1]['serial_number'].unique())
    n_failed_serial_numbers = len(failed_serial_numbers)
    training_serial_numbers = random.sample(failed_serial_numbers, int(round(n_failed_serial_numbers * 0.7)))

    test_serial_numbers = [item for item in failed_serial_numbers if item not in training_serial_numbers]

    training_datasets = [get_last_100(df, serial_number) for serial_number in training_serial_numbers]
    testing_datasets = [get_last_100(df, serial_number) for serial_number in test_serial_numbers]

    training_dataset = pd.concat(training_datasets)
    testing_dataset = pd.concat(testing_datasets)

    training_dataset.to_csv('training_harddrive.csv', index=False)
    testing_dataset.to_csv('testing_harddrive.csv', index=False)
