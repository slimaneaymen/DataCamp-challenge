import os
from glob import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_file_list_from_dir(*, path='.', datadir='data'):
    data_files = sorted(glob(os.path.join(path,datadir, "*.csv")))
    return data_files

def clean_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        df.drop(["reference_video", "start_point"], axis=1, inplace=True)
    except KeyError:
        df = pd.read_csv(file_path, sep=";")
        df.drop(["reference_video", "start_point"], axis=1, inplace=True)
    #print(df.head())
    return df

def get_data_targets(file_path):
    """
    returns data, targets. data is a list of sequences, targets a list of couples
    It splits the dataset whenever it is Djokovic's turn to play
    """
    df = clean_csv(file_path)
    df_djoko = df[df["play_djoko"]==1]
    targets = df_djoko[["targetx_ball", "targety_ball"]].to_numpy()

    ind_djoko = list(df_djoko.index)
    n = len(ind_djoko)
    sequences = []
    MAX_SEQUENCE_LENGTH = 10 
    for i in ind_djoko:
        sequence = df.iloc[:i,:]
        sequence = sequence.to_numpy()
        sequences.append(sequence)

    #data = np.array(data).reshape(n,-1,13)
    sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, value=-100)
    #print(data.shape)
    return sequences, targets


if __name__ == "__main__":
    #files_names = get_file_list_from_dir(path=".", datadir="data")
    #df = pd.concat((pd.read_csv(f) for f in files_names),ignore_index=True).drop(['reference_video', 'start_point'], axis=1)
    #df.to_csv('./dataset.csv') 
    file_path = "./data/terence1.csv"
    df = clean_csv(file_path)
    print(df.head())