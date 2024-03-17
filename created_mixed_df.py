import argparse
from config import *
from utils import *
import pickle
import h5py
from operator import itemgetter
"""
The goal of this script is to create a dataframe that can be used both for the metadata and for the time series data
"""

def create_save_df(args):

    # First step: loading the metadata into a pandas df
    def generator_function():
        with open(metadata_pickle_file_path, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    yield data
                except EOFError:
                    break
    for item in generator_function():
        df = item
    
    
    df["read_id"] = df.index
    print(f'Initially there are {df["read_id"].nunique()} read_ids')

    # Second step: loading the time series data into dictionaries of reads

    fold_dict = CV_FOLDS[args.fold]
    train_pos = load_multiple_ds_metadata(itemgetter(*fold_dict['train_pos'])(SAMPLE_PATHS), args.train_files_limit)
    train_neg = load_multiple_ds_metadata(itemgetter(*fold_dict['train_neg'])(SAMPLE_PATHS), args.train_files_limit)
    test_pos = load_multiple_ds_metadata(itemgetter(*fold_dict['test_pos'])(SAMPLE_PATHS), args.test_files_limit)
    test_neg = load_multiple_ds_metadata(itemgetter(*fold_dict['test_neg'])(SAMPLE_PATHS), args.test_files_limit)
    _train_reads = {1: train_pos, 0: train_neg}
    _test_reads = {1: test_pos, 0: test_neg}

    # Creating an intermediairy dataset that will contain info such as: path, label, and train/validation flag
    additional_info = create_labels_df_from_data_dicts([(_train_reads, "train"), (_test_reads, "test")])
    print(f'read_ids in the additional_info dataframe : {additional_info["read_id"].nunique()}')

    # Enriching the initial dataframe with the above information
    df = pd.merge(df, additional_info, on='read_id', how="right")
    df = df.reset_index()

    df_train = df[df['train_test_flag'] == 'train'].reset_index()
    df_test = df[df['train_test_flag'] == 'test'].reset_index()

    read_id_to_row_number_train = [(row["read_id"],row["path"],row["label"]) for counter, (index, row) in enumerate(df_train.iterrows())]
    read_id_to_row_number_test = [(row["read_id"],row["path"],row["label"]) for counter, (index, row) in enumerate(df_test.iterrows())]

    def get_signal_array(path, read_id):
        try:
            with h5py.File(path, 'r') as f:
                if 'Raw' in f:  # Single read file
                    signal = f.get('Raw/Reads/read_' + read_id + '/Signal')
                else:  # Multi read file
                    signal = f.get('read_'+read_id).get('Raw/Signal')
                return signal[:]
        except (AttributeError, FileNotFoundError) as e:
            path = path.replace("_pass", "_fail")
            with h5py.File(path, 'r') as f:
                if 'Raw' in f:  # Single read file
                    signal = f.get('Raw/Reads/' + read_id + '/Signal')
                else:  # Multi read file
                    signal = f.get('read_'+read_id).get('Raw/Signal')
                return signal[:]
            
    list_valid_reads = []
    list_valid_paths = []

    # The goal here is to save only the read_ids that have a valid signal
    for _ in [read_id_to_row_number_train, read_id_to_row_number_test]:
        for read_id, path, label in _:
            try:
                signal = get_signal_array(path,read_id)
                list_valid_reads.append(read_id)
                list_valid_paths.append(path)
            except:
                pass

    reliable_paths_df = pd.DataFrame()
    reliable_paths_df["path"] = list_valid_paths
    reliable_paths_df["read_id"] = list_valid_reads

    # Doing an inner join because we only want to consider the paths with a valid signal
    df = pd.merge(df, reliable_paths_df, on=['read_id','path'], how="inner")

    df['train_test_flag'] = df['train_test_flag'].replace({'test': 1, 'train': 0})
    df['is_valid'] = df['train_test_flag'].astype('bool')

    print(f'Final dataframe shape: {df.shape}')
    print(f'There are {df["read_id"].nunique()} read_ids in the final dataframe')
    
    df.to_pickle(f'mixed_datasets_new/mixed_df_fold_{args.fold}.pkl')

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Saving mixed df.')
    parser.add_argument('--fold', default=2, type=int, help='cvfold')
    parser.add_argument('--train_files_limit', default=300000, type=int)
    parser.add_argument('--test_files_limit', default=100000, type=int)
    args = parser.parse_args()
    create_save_df(args=args)