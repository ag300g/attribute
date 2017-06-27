"""
Some useful functions to manage file on the server
"""

import os, errno
import subprocess
import pandas as pd
import yaml
import time
import os
from shutil import copyfile

def archiveFile(path, fileName):
    """
    TimeStamp and archive the specified file
    """
    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    if(os.path.isfile(path + fileName)):
        newFileName = fileName + timestamp + '.h5'
        if(os.path.isfile(path + newFileName)):
            newFileName = fileName + '_2_' + timestamp + '.h5'
        copyfile(path + fileName, path + newFileName)
        print("File archived at: " + path + newFileName)
    else:
        print("No archive needed")
    return

def loadSettingsFromYamlFile(filePath):
    """
    Load settings from a yaml file specified by the filepath
    """
    with open(filePath, 'r') as f:
        settings = yaml.load(f)
    return settings

def store_df_in_named_file(df, name, downloadLocally = False):
    """
    Stores Pandas DataFrame df in h5 file with name "name"
    """


    silentremove(ensure_suffix(name, '.h5'))
    with pd.HDFStore(ensure_suffix(name, '.h5'), complevel=9,
                     complib='blosc') as store:
        store['df'] = df


    """
    df.to_csv(name + ".txt", sep = '\t', encoding = 'utf-16')
    """

    if downloadLocally:
        print("Downloading " + str(ensure_suffix(name, '.h5')))
        subprocess.call(["sz",ensure_suffix(name, '.h5')])


def get_df_from_named_file(name, df_name='df'):
    """
    Gets Pandas DataFrame name from h5 file with name "df_name"
    """
    
    

    with pd.HDFStore(ensure_suffix(name, '.h5')) as store:
        return store[df_name]


    """
    df = pd.read_csv(name + ".txt", sep = '\t', encoding = 'utf-16')
    """


def silentremove(filename):
    """
    Silently removes file. Usually h5 file.
    """
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured





def ensure_suffix(string, suffix):
    """
    Ensures that a file name "string" has a suffix "suffix".
    If it doesn't have it, the function adds suffix to the file name "string"
    """
    if string.endswith(suffix):
        return string
    else:
        return string + suffix
