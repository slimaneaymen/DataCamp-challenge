import os
from glob import glob
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

def get_df_data(datadir='data', service_row=None):
    L= []
    onlyfiles = [f for f in listdir(datadir) if isfile(join(datadir, f))]
    onlyfiles.remove('.DS_Store')
    for name in onlyfiles:
        df_sequence = pd.read_csv(join('data',name), sep=';',index_col=False)
        if df_sequence.iloc[-1, 0]==0:
            df_sequence = df_sequence.drop(index = df_sequence.shape[0]-1)
        if df_sequence.iloc[0, 0]==1:
            df_sequence = pd.concat([service_row, df_sequence], axis=0)
        
        L.append(df_sequence)
    concat_df = pd.concat(L, axis=0, ignore_index=True) 
    return concat_df