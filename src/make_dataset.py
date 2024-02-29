import pandas as pd
import os

def get_files_labels(dataset, label):
    """
    Traverse a directory using os.walk, collect file names, and create a pandas DataFrame.
    Parameters:
    dataset : str; The path of the directory to traverse.

    Returns: pandas.DataFrame; A DataFrame containing two columns: "File Name"
    (full path of each file) and "Label" (passed as the second argument).
    """
    file_names = []

    for root, _, files in os.walk(dataset,topdown=True):
        for file in files:
            file_path = os.path.join(root, file)
            file_names.append(file_path)

    l = len(file_names)
    labels = [label]*l

    df = pd.DataFrame({'file_name': file_names, 'label': labels})
    return df

def make_dataset(dataset1, dataset2, labelset1 = 1, labelset2 = 0):
    '''
    Given two datasets, both for either training or testing, concatenates the two. Additionally,
    a label is added - by default it is 1 for dataset1 and 0 for dataset2, but these can be
    passed into the function

    Params:
    dataset1: path to the folder containing the first dataset
    dataset2: path to the folder containing the second dataset
    labelset1: label for the first dataset, set to 1 by default
    labelset2: label for the second dataset, set to 0 by default

    Returns:
    dataset: concatenated dataset containing links to particular files, along with appropriate labels, stored
    in a dataframe
    '''


    data1 = get_files_labels(dataset1, labelset1)
    data2 = get_files_labels(dataset2, labelset2)

    df = pd.concat([data1, data2], axis=0, ignore_index=True)

    return df
