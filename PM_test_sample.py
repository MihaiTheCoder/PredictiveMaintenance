import pandas
from process_input import create_feature_records

def read_file_line_by_line_into_list(file_path):
    with open(file_path) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    return content

def create_sample_with_RUL(file):
    original_df = create_feature_records.create_dataset_from_input(file)
    truth_df = pandas.DataFrame({'RULL': pandas.Series(read_file_line_by_line_into_list('PM_truth.txt'), dtype='int32')})
    truth_df['id'] = truth_df.index + 1
    truth_df = truth_df.reset_index(drop=True)
    original_df = original_df.merge(truth_df, on='id')
    counts_df = original_df.groupby(['id', 'RULL']).count()
    ccc_df = pandas.DataFrame({'count': counts_df['cycle']})
    ccc_df = ccc_df.reset_index()
    ccc_df['id'] = ccc_df.index + 1
    ccc_df = ccc_df.reset_index(drop=True)
    ccc_df['total_rul'] = ccc_df['RULL'] + ccc_df['count']
    ccc_df = ccc_df.drop(['RULL', 'count'], axis=1)
    original_df = original_df.merge(ccc_df, on='id')
    original_df['RUL'] = original_df['total_rul'] - original_df['cycle']
    sample_test_df = original_df.drop(['RULL', 'total_rul'], axis=1)

    return sample_test_df

if __name__ == '__main__':
    sampled_test_df = create_sample_with_RUL('PM_test.txt')
    sampled_test_df.to_csv("PM_test_sample.txt", sep=' ', index=False, header=False)
    #sampled_test_df.to_csv("PM_test_sample.csv")