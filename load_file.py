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
from zip_codes import ZC # zip code database


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
	'''reports the number (and percentage) of unique elements in a pd.Series.
	used to determine when a column should be converted to type category.
	string columns with than 50% unique elements will see a memory savings upon 
	conversion to category'''
	name = pd_series.name
	u_len = len(pd_series.unique())
	pd_len = len(pd_series)
	print( '{0} has {1} unique elements ({2}%)'.format(name, u_len, 100*u_len/pd_len) )
	return

def cat_cleaner(cat):
	'''
	given a categorical string, this function splits, cleans, sorts and 
	recombines with spaces in between category values
	
	cat: input. string of categories separated by commas. may have misspellings, nan's or other 
		known bad features. example:
			cat = 'IAB1,IAB1_1,IAB1_6,IAB2,IAB3,IAB3_4,IAB5,IAB6,IAB7,IAB7_1,IAB7_32,IAB7_44,IAB8,IAB8_17,IAB9,IAB9_1,IAB9_10,IAB9_11,IAB9_5,IAB9_7,-1,IAB7_,I,,,'
		this cat has a -1, a typo (IAB7_) and empty entries (,,,)
	'''
	# list of known bad categories / typos to be removed
	bad_cats = ['-1', 'IAB', 'IAB7_', 'IA', 'I', ' ', 'nan', 'NaN']
	
	# split into individual elements
	cat_list = cat.split(',')

	#correct common misspellings
	cleancat = ['IAB7' if s=='IAB7_' else s for s in cat_list]

	# remove known bad elements from cat_list, including empty strings
	cleancat = sorted(list(set(cat_list)-set(bad_cats)))
	cleancat = list(filter(None, cleancat))

	# convert back to a string of elements separated by spaces
	return ' '.join(cleancat)

def segment_cleaner(seg):
	''' cleans the segment column '''
	seg = str(seg) # force string conversion
	seg = seg.translate( {ord(c): None for c in '[,]'} ) # drop [] and commas
	return ' '.join(sorted(seg.split(' '))) # split, sort and recombine with spaces

def load_data(data_dir=r'C:\PythonBC\RootData', fname='2019-04-27.csv', Verbose=False, all_cols=True, sub_cols=['geo_zip'], **kwargs):
	'''loads a CSV or ZIP file to a pandas dataframe. note: can only work with a single compressed file, not a bundle of compressed files
	
	input parameters:

	data_dir: directory of fname
	fname: name of file to be read (don't include directory info)
	Verbose: should we print the memory size of the dataframe?
	all_cols: set True to load all columns from fname. set False if you only
		want a subset.
	sub_cols: only implemented if all_cols==False. sub_cols is a list of 
		strings of column names to be loaded. must be comprised of the elements 
		of the mycols list (defined below). note: sub_cols overrides the hard-
		coded <unwanted> list.
	
	'''

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
	
	#we never want these columns, drop them from <mycols>
	unwanted = ['auction_id', 'platform_os', 'day', 'day_of_week', 'month', 'year', 'hour', 'creative_size', 'app_bundle']
	mycols = [ele for ele in mycols if ele not in unwanted]

	if not all_cols:
		# extract the user-defined columns from <mycols> if they are also in <sub_cols>
		mycols = [ele for ele in mycols if ele in sub_cols]
	
	# specify datatypes for each column
	mydtype = {'auction_id': str,
		'inventory_source': 'category',
		'app_bundle': 'category',
		'category': 'category',
		'inventory_interstitial': np.bool,
		'geo_zip': float,
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
	if all_cols: # drop the unnamed 1st column ('')
		df = pd.read_csv(os.path.join(data_dir, fname), usecols=mycols[1:], dtype=mydtype, sep=',', engine='c', **kwargs)
	else:
		df = pd.read_csv(os.path.join(data_dir, fname), usecols=mycols, dtype=mydtype, sep=',', engine='c', **kwargs)
		
	if Verbose:
		print('memory size after loading:', mem_usage(df))

	# do some dtype conversions on columns to save memory, clean data
	
	if 'bid_timestamp_utc' in mycols:
		# manually add the UTC timezone to this column
		df.bid_timestamp_utc = pd.to_datetime( df.bid_timestamp_utc, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce', utc=True )
	
	if 'geo_zip' in mycols:
		df.geo_zip = df.geo_zip.apply(fix_zipcode)
		df.geo_zip = df.geo_zip.astype('category')

	if 'category' in mycols: #remove bad categories
		df.category = df.category.apply( cat_cleaner )
	
	if 'platform_carrier' in mycols: # fix bad platform_carriers
		df.platform_carrier = df.platform_carrier.replace(['-1'], np.nan)
	
	if 'platform_device_make' in mycols:
		df.platform_device_make = df.platform_device_make.replace( [' ', '-1'], np.nan )
	
	if 'platform_device_model' in mycols:
		df.platform_device_model = df.platform_device_model.replace([' ', '-1'], np.nan)
	
	if 'platform_device_screen_size' in mycols:
		df.platform_device_screen_size = df.platform_device_screen_size.replace(['nan', ' ', '-1', 'NaN', 'UNKNOWN'], value=np.nan)

	if 'segments' in mycols:
		df.segments = df.segments.apply( segment_cleaner )
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
	'''loads a Parquet file that was saved using temp_save
	faster than re-loading and re-processing the csv'''
	_table = (pq.ParquetFile(fname)
			.read(use_pandas_metadata=True))
	df = _table.to_pandas(strings_to_categorical=True) # force strings to categoricals to save memory
	return df

def load_wrapper(data_dir, fname, **kwargs):
	zc = ZC('') # initialize zip code class
	df = load_data(fname=fname, data_dir=data_dir, **kwargs)
	df = df.dropna(subset=['geo_zip']) # drop NaN values of zip codes

	# use zip code to calculate things
	df['tz'] = zc.zip_to_tz_2(df.geo_zip) # get timezone
	df['tz'] = df.tz.astype('category')
	df['state'] = zc.zip_to_state_2(df.geo_zip) # get state
	df['state'] = df.state.astype('category')
	#df = df.dropna( subset=['tz']) # drop rows that don't have a timezone
	df = zc.shift_tz_wrap(df, style='careful') # get local times
	df['hour'] = zc.local_hour(df) # create new column of local hour
	return df
#%% example usage

#fname = '2019-04-27.csv'
#data_dir = r'C:\PythonBC\RootData'
#df = load_wrapper(fname=fname, data_dir=data_dir, nrows=50000)

