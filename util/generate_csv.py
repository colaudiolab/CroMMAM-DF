import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os
import decord

def load_args():
    parser = argparse.ArgumentParser(description='Generate_data_csv')
    parser.add_argument('--root-dir', default=None, help='video directory')
    parser.add_argument('--csv-name', default=None, help='saved csv file name')
    parser.add_argument('--target-dir', default='/mnt/200ssddata2t/yejianbin/DFDC', help='csv directory')
    parser.add_argument('--label', default=None, help='label of video')
    args = parser.parse_args()
    return args

def main(IS_SPLIT=0):
    args = load_args()
    real_file_name_list = []
    fake_file_name_list = []
    real_labels = [] 
    fake_labels = []
    
    for path in tqdm(Path(args.root_dir).rglob("*.mp4")):
        relpath = os.path.relpath(path, args.root_dir)
        try:
            decord_vr = decord.VideoReader(str(path), num_threads=1)
            duration = len(decord_vr)
        except Exception as err:
            print(err)
            continue
        
        if 'Real' in str(path):
            real_labels.append(0)
            real_file_name_list.append(relpath)
        else :
            fake_labels.append(1)
            fake_file_name_list.append(relpath)

    # labels = [args.label]*len(file_name_list)
    file_name_list = real_file_name_list + fake_file_name_list
    labels = real_labels + fake_labels
    if IS_SPLIT:
        train, test = split_dataset(file_name_list, labels)
        train_dataframe = pd.DataFrame({'path':train[0],'label':train[1]}, columns=['path','label'])
        # val_dataframe = pd.DataFrame({'path':val[0],'label':val[1]}, columns=['path','label'])
        test_dataframe = pd.DataFrame({'path':test[0],'label':test[1]}, columns=['path','label'])

        train_dataframe.to_csv(os.path.join(args.target_dir, "train.csv"),  index=False, sep=' ', header=False)
        # val_dataframe.to_csv(os.path.join(args.target_dir, "val.csv"),  index=False, sep=' ', header=False)
        test_dataframe.to_csv(os.path.join(args.target_dir, "test.csv"),  index=False, sep=' ', header=False)
    else:
        dataframe = pd.DataFrame({'path':real_file_name_list,'label':real_labels}, columns=['path','label'])
        dataframe.to_csv(os.path.join(args.target_dir, "real.csv"),  index=False, sep=' ', header=False)
        dataframe = pd.DataFrame({'path':fake_file_name_list,'label':fake_labels}, columns=['path','label'])
        dataframe.to_csv(os.path.join(args.target_dir, "fake.csv"),  index=False, sep=' ', header=False)

def split_dataset(x, y):
    from sklearn.model_selection import train_test_split

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    # return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    # 没有验证集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    
    main(IS_SPLIT=0)