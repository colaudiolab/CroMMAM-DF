from __future__ import print_function
import os

from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_files_in_dir(dir):
    import decord
    files = []
    labels = []
    duration_list = []
    for path in tqdm(Path(dir).rglob("*.mp4")):
        try:
            decord_vr = decord.VideoReader(str(path), num_threads=1)
            duration = len(decord_vr)
        except Exception as err:
            print(err)
            continue

        if 'Real' in str(path):
            labels.append(0)
        else:
            labels.append(1)

        duration_list.append(duration)
        relpath = os.path.relpath(path, dir)
        files.append(relpath)
    return files, labels, duration_list


def generate_all_dataset(root_dir):
    data_dir = os.path.join(root_dir)
    basename = os.path.basename(data_dir)
    path = os.path.join(root_dir, f'../{basename}.csv')
    print(path)

    x_data, y_data, duration_list = get_files_in_dir(data_dir)
    x_data = [basename+'/'+x for x in x_data]
   
    train_dataframe = pd.DataFrame({'path':x_data,'label':y_data, 'duration':duration_list}, columns=['path','label','duration'])

    train_dataframe.to_csv(path,  index=False, sep=' ', header=False)

def split_FR_dataset(n):
    root_dir = '/mnt/200ssddata2t/yejianbin/DFDC/processed'
    fake_csv_path = os.path.join(root_dir, 'fake.csv')
    real_csv_path = os.path.join(root_dir, 'real.csv')
    # test_fake_path = os.path.join(root_dir, 'test_fake.csv')
    # test_real_path = os.path.join(root_dir, 'test_real.csv')
    # val_fake_path = os.path.join(root_dir, 'val_fake.csv')
    # val_real_path = os.path.join(root_dir, 'val_real.csv')
    # train_fake_path = os.path.join(root_dir, 'train_fake.csv')
    # train_real_path = os.path.join(root_dir, 'train_real.csv')
    train_path = os.path.join(root_dir, f'train-{n}.csv')
    test_path = os.path.join(root_dir, f'test-{n}.csv')

    # train_size = 0.8
    # val_size = 0.1
    # test_size = 0.1

    fake_data = pd.read_csv(fake_csv_path, sep=' ')
    real_data = pd.read_csv(real_csv_path, sep=' ')

    x_fake_data = fake_data.iloc[:, 0].values
    y_fake_data = fake_data.iloc[:, 1].values
    d_fake_data = fake_data.iloc[:, 2].values
    x_real_data = real_data.iloc[:, 0].values
    y_real_data = real_data.iloc[:, 1].values
    d_real_data = real_data.iloc[:, 2].values

    x_train_fake, x_test_fake, y_train_fake, y_test_fake, d_train_fake, d_test_fake = train_test_split(x_fake_data, y_fake_data, d_fake_data, test_size=0.2, shuffle=True, random_state=n)
    x_train_real, x_test_real, y_train_real, y_test_real, d_train_real, d_test_real = train_test_split(x_real_data, y_real_data, d_real_data, test_size=0.2, shuffle=True, random_state=n)
    x_train_fake = x_train_fake[:8000]
    y_train_fake = y_train_fake[:8000]
    d_train_fake = d_train_fake[:8000]
    x_train_real = x_train_real[:8000]
    d_train_real = d_train_real[:8000]
    y_train_real = y_train_real[:8000]

    x_test_fake = x_test_fake[:2000]
    y_test_fake = y_test_fake[:2000]
    d_test_fake = d_test_fake[:2000]
    x_test_real = x_test_real[:2000]
    y_test_real = y_test_real[:2000]
    d_test_real = d_test_real[:2000]
    
    x_train = np.concatenate((x_train_fake, x_train_real))
    y_train = np.concatenate((y_train_fake , y_train_real))
    d_train = np.concatenate((d_train_fake , d_train_real))
    x_test = np.concatenate((x_test_fake , x_test_real))
    y_test = np.concatenate((y_test_fake , y_test_real))
    d_test = np.concatenate((d_test_fake , d_test_real))

    # x_val_fake, x_test_fake, y_val_fake, y_test_fake = train_test_split(x_test_fake, y_test_fake, test_size=0.5, random_state=42)
    # x_val_real, x_test_real, y_val_real, y_test_real = train_test_split(x_test_real, y_test_real, test_size=0.5, random_state=42)

    # train_fake_dataframe = pd.DataFrame({'path':x_train_fake,'label':y_train_fake}, columns=['path','label'])
    # train_real_dataframe = pd.DataFrame({'path':x_train_real,'label':y_train_real}, columns=['path','label'])
    # val_fake_dataframe = pd.DataFrame({'path':x_val_fake,'label':y_val_fake}, columns=['path','label'])
    # val_real_dataframe = pd.DataFrame({'path':x_val_real,'label':y_val_real}, columns=['path','label'])
    # test_fake_dataframe = pd.DataFrame({'path':x_test_fake,'label':y_test_fake}, columns=['path','label'])
    # test_real_dataframe = pd.DataFrame({'path':x_test_real,'label':y_test_real}, columns=['path','label'])

    # train_fake_dataframe.to_csv(train_fake_path,  index=False, sep=' ', header=False)
    # train_real_dataframe.to_csv(train_real_path,  index=False, sep=' ', header=False)
    # val_fake_dataframe.to_csv(val_fake_path,  index=False, sep=' ', header=False)
    # val_real_dataframe.to_csv(val_real_path,  index=False, sep=' ', header=False)
    # test_fake_dataframe.to_csv(test_fake_path,  index=False, sep=' ', header=False)
    # test_real_dataframe.to_csv(test_real_path,  index=False, sep=' ', header=False)

    train_dataframe = pd.DataFrame({'path':x_train,'label':y_train, 'duration':d_train}, columns=['path','label','duration'])
    test_dataframe = pd.DataFrame({'path':x_test,'label':y_test, 'duration':d_test}, columns=['path','label','duration'])

    train_dataframe.to_csv(train_path,  index=False, sep=' ', header=False)
    test_dataframe.to_csv(test_path,  index=False, sep=' ', header=False)

def split_dataset():
    root_dir = '/mnt/200ssddata2t/yejianbin/DFDC/processed/Fake'
    data_dir = os.path.join(root_dir)
    test_path = os.path.join(root_dir, '../fake-test.csv')
    # val_path = os.path.join(root_dir, 'val.csv')
    train_path = os.path.join(root_dir, '../fake-train.csv')

    # train_size = 0.8
    # val_size = 0.1
    # test_size = 0.1
    x_data, y_data, duration_list = get_files_in_dir(data_dir)
    # y_data = [0] * len(x_data)

    x_train, x_test, y_train, y_test, d_train, d_test = train_test_split(x_data, y_data, duration_list, test_size=0.2, shuffle=True, random_state=1)
    # x_train = x_train[:8000]
    # y_train = y_train[:8000]
    # d_train = d_train[:8000]
    # x_test = x_test[:2000]
    # y_test = y_test[:2000]
    # d_test = d_test[:2000]

    # x_val, x_test, y_val, y_test, d_val, d_test = train_test_split(x_test, y_test, d_test, test_size=0.5, random_state=42)

    train_dataframe = pd.DataFrame({'path':x_train,'label':y_train, 'duration':d_train}, columns=['path','label','duration'])
    # val_dataframe = pd.DataFrame({'path':x_val,'label':y_val, 'duration':d_val}, columns=['path','label','duration'])
    test_dataframe = pd.DataFrame({'path':x_test,'label':y_test, 'duration':d_test}, columns=['path','label','duration'])

    train_dataframe.to_csv(train_path,  index=False, sep=' ', header=False)
    # val_dataframe.to_csv(val_path,  index=False, sep=' ', header=False)
    test_dataframe.to_csv(test_path,  index=False, sep=' ', header=False)

if __name__ == '__main__':
    # split_dataset()
    # for i in range(1,11):
    #     split_FR_dataset(i)
    for typ in ['BW', 'CC', 'CS', 'GB', 'GNC', 'JPEG']:
        root_dir = f'/mnt/200ssddata2t/yejianbin/FakeAVCeleb/csv_files-2400/perturbation/{typ}'
        for item in os.listdir(root_dir):  
            item_path = os.path.join(root_dir, item)  
            # 如果是目录，递归遍历  
            if os.path.isdir(item_path):  
                generate_all_dataset(item_path)
    
    # generate_all_dataset(root_dir)
