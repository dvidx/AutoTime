"""[summary]

Returns:
    [type]: [description]
"""
import time
import numpy as np
import pandas as pd
import os


def stat_dist(dataset, cols_index, n_steps):
    """[summary]

    Args:
        dataset ([type]): [description]
        cols_index ([type]): [description]
        n_steps ([type]): [description]

    Returns:
        [type]: [description]
    """
    mean_store = []
    std_store = []
    min_store = []
    max_store = []

    val_list = list()

    for col_index in cols_index:
        # name = dataset.columns[col_index]
        name = col_index
        for index, row in dataset.iterrows():
            if index >= n_steps:
                val_list = dataset[name][index-n_steps+1:index+1]

                mean = np.mean(val_list)
                std = np.std(val_list)
                min = np.min(val_list)
                max = np.max(val_list)

            else:
                mean = row[name]
                std = 0
                min = row[name]
                max = row[name]

            mean_store.append(mean)
            std_store.append(std)
            min_store.append(min)
            max_store.append(max)

        df2 = pd.DataFrame({f'{name}_mean': mean_store,
                            f'{name}_std': std_store,
                            f'{name}_min': min_store,
                            f'{name}_max': max_store
                            })
        dataset = dataset.merge(df2, left_index=True, right_index=True)

    return dataset



def stat_main(org_dataset, dist_steps, out_path):
    # copy and format dataset
    dataset = org_dataset.copy()
    dataset.reset_index(drop=True, inplace=True)
    # dataset.astype('float32')

    # Get all columns that are numerical (not binary or categorical)
    # and append them to a list
    # cols_index = [0, 2, 6] # ToDo - Make adaptable 
    cols_index = dataset._get_numeric_data().columns

    # variable that defines the time of these distributions
    n_steps = dist_steps 
    #dataset.iloc[ : , cols]

    # Function call
    modified_df = stat_dist(dataset, cols_index, n_steps)



    # Output copy
    pathname = "/files/"
    # ts = time.gmtime()
    # iteration = time.strftime("%Y_%m_%d", ts)
    filename = "_stat_dist.csv"
    dirpath = "{}{}_{}".format(out_path, pathname, filename) # iteration,
    modified_df.to_csv(dirpath, index=False, sep=",")

    modified_df = modified_df._get_numeric_data()
    
    return modified_df