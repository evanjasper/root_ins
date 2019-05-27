# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:55:18 2019

@author: Greg Smith
"""

#%% import stuff
import pandas as pd
import numpy as np
import time, os, glob
import pyarrow.parquet as pq

def mem_usage(pd_obj):
    '''calculates the memory footprint of a pandas object'''
    if isinstance(pd_obj,pd.DataFrame):
        usage_b = pd_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pd_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.3f} MB".format(usage_mb)

def fix_zipcode(s):
    ''' fix zip codes: '43212.0' --> '43212', '-1.0' --> np.nan, nan --> np.nan '''
    fixed = str(s).split('.')[0]
    if len(fixed) == 5:
        return fixed
    else:
        return np.nan

def fix_hour(pd_series):
    '''fix hour column: 'HH:00' (string) --> HH (int8)'''
    def fix_string(s):
        '''drops the : and trailing zeroes ('22:00' --> '22')'''
        return str(s).split(':')[0]
    reduced =  pd_series.apply(fix_string)
    return reduced.astype('int8')

def unique_counter(pd_series):
    '''reports the number (and percentage) of unique elements in a pd.Series'''
    name = pd_series.name
    u_len = len(pd_series.unique())
    pd_len = len(pd_series)
    print( '{0} has {1} unique elements ({2}%)'.format(name, u_len, 100*u_len/pd_len) )
    return

def load_data(data_dir=r'C:\PythonBC\Project\Data', fname='2019-04-27.csv', nrows='All', Verbose=False):
    '''loads a CSV or ZIP file to a pandas dataframe. note: can only work with a single compressed file, not a bundle of compressed files'''

    mycols = ['',
          'auction_id',
          'inventory_source',
          'app_bundle',
          'category',
          'inventory_interstitial',
          'geo_zip',
          'platform_bandwidth',
          'platform_carrier',
          'platform_os',
          'platform_device_make',
          'platform_device_model',
          'platform_device_screen_size',
          'rewarded',
          'bid_floor',
          'bid_timestamp_utc',
          'hour',
          'day',
          'day_of_week',
          'month',
          'year',
          'segments',
          'creative_type',
          'creative_size',
          'spend',
          'clicks',
          'installs']

    mydtype = {'auction_id': str,
        'inventory_source': 'category',
        'app_bundle': 'category',
        'category': 'category',
        'inventory_interstitial': np.bool,
        'geo_zip': str,
        'platform_bandwidth': 'category',
        'platform_carrier': 'category',
        'platform_os': 'category',
        'platform_device_make': 'category',
        'platform_device_model': 'category',
        'platform_device_screen_size': 'category',
        'rewarded': np.bool,
        'bid_floor': np.float64,
        'bid_timestamp_utc': str,
        'hour': str,
        'day': 'int8',
        'day_of_week': 'category',
        'month': 'int8',
        'year': 'int16',
        'segments': 'category',
        'creative_type': 'category',
        'creative_size': 'category',
        'spend': np.float64,
        'clicks': np.bool,
        'installs': np.bool}
    
    #load data
    if nrows == 'All':
        df = pd.read_csv(os.path.join(data_dir, fname), usecols=mycols[1:], dtype=mydtype)
    elif int(nrows)>0: 
        df = pd.read_csv(os.path.join(data_dir, fname), usecols=mycols[1:], dtype=mydtype, nrows=int(nrows))
    
    if Verbose:
        print('memory size after loading:', mem_usage(df))

    # do some dtype conversions on columns to save memory, clean data
    df.bid_timestamp_utc = pd.to_datetime( df.bid_timestamp_utc, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce', utc=True ) # fix timestamp
    df.geo_zip = df.geo_zip.apply(fix_zipcode)
    df.hour = fix_hour(df.hour)
    df.geo_zip = df.geo_zip.astype('category')
    
    # drop useless columns
    df = df.drop(columns = ['auction_id']) # has no predictive power
    df = df.drop(columns = ['platform_os']) # only consists of 'Android' or '-1'
    df = df.drop(columns = ['day', 'month', 'year', 'hour']) #these numbers correspond to UTC, not useful
    if Verbose:
        print('memory size after downcasting:', mem_usage(df))
    return df

def load_all_files(data_dir, type='csv'):
    '''load all CSVs or ZIPs in data_dir into a single pd.dataframe'''
    if type == 'csv':
        all_files = [os.path.basename(x) for x in glob.glob(os.path.join(data_dir, '*.csv'))]
    elif type == 'zip':
        all_files = [os.path.basename(x) for x in glob.glob(os.path.join(data_dir, '*.zip'))]
    df_from_each_file = (load_data(data_dir=data_dir, fname=f, nrows='All', Verbose=False) for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    return df

def temp_save(df, fname='df.gzip'):
    '''saves a pd.df to disk for later use using Parquet'''
    df.to_parquet(fname, engine='pyarrow')
    
def temp_load(fname='df.gzip'):
    '''loads a Parquet file that was saved using temp_save'''
    _table = (pq.ParquetFile(fname)
            .read(use_pandas_metadata=True))
    df = _table.to_pandas(strings_to_categorical=True) # force strings to categoricals to save memory
    return df
#%% example usage
'''
fname = '2019-04-27.csv'
fname_zip = '2019-04-27.zip'
data_dir = r'C:\PythonBC\Project\Data'

start = time.time()
df = load_data(fname=fname, data_dir=data_dir, Verbose=True)
#df = load_all_files(data_dir)
end = time.time()
print('time to load file: {0:3.3f} seconds'.format(end-start))

# get an idea of how much memory we are using
print(df.memory_usage(deep=True))
print('size in memory:', mem_usage(df)) #173.278 MB

# compare to default settings (raises an error)
#df2 = pd.read_csv(data_dir + '\\' + fname)
#print('size in memory:', mem_usage(df2)) # 1478 MB
'''
#%% try to plot things
#df['platform_bandwidth'].value_counts().plot(kind='bar')