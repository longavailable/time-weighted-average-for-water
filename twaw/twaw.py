# -*- coding: utf-8 -*-
'''
* Updated on 2023/03/06
* python3
**
* Calculate time(hour)-weighted daily average
'''

from datetime import datetime, timedelta
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
import pandas as pd
import numpy as np

def addDays(startDate, n, skip_leap_days=False):
	'''A function to operate addition for a date.
	
	Parameters:
		startDate:
			Type: datetime.date
		n:
			Type: int
		skip_leap_days:
			Type: boolean
			Default: False
	Returns:
		A datetime.date object
	'''
	d = startDate + timedelta(n)
	if skip_leap_days and d.month==2 and d.day==29: return None
	return d

def daterange(startDate, endDate, skip_leap_days=False):
	'''A function to abstract the iteration over the range of dates.
	
	Parameters:
		startDate:
			Type: datetime.date
		endDate:
			Type: datetime.date
		skip_leap_days:
			Type: boolean
			Default: False
	Returns:
		A list
	'''
	
	'''
	for n in range(int((endDate - startDate).days)):
		yield startDate + timedelta(n)
	'''
	
	'''
	# single thread version
	dates = []
	for n in range(int((endDate - startDate).days)):
		d = addDays(startDate, n, skip_leap_days)
		dates = dates + [d]
	return dates
	'''
	# parallel
	pool = ThreadPool(4)
	dates = pool.starmap(addDays, zip(repeat(startDate), list(range(int((endDate - startDate).days))), repeat(skip_leap_days)))
	pool.close()
	pool.join()
	return list(filter(lambda d: d is not None, dates))

def filterByTime(pdf, d0, d1, timeHeader):
	'''Filter subset by datetime specified its header
	
	Parameters:
		pdf: source data to filter
			Type: pandas.DataFrame object
		d0, d1: start date and end date
			Type: datetime.datetime or datetime.date object
		timeHeader: label/header of time column
			Type: string
	return:
		pandas.DataFrame object
	'''
	expr = '"%s"<=%s<="%s"' % (d0.strftime('%Y-%m-%d 00:00:00'), timeHeader, d1.strftime('%Y-%m-%d 00:00:00'))
	return pdf.query(expr)

def interpolateByTime(pdf0, pdf1, itemHeader, timeHeader):
	'''Interpolate data at 0:00
	
	Parameters:
		pdf0: data befor 0:00
			Type: pandas.DataFrame object
		pdf1: data after 0:00
			Type: pandas.DataFrame object
		itemHeader: label/header of variable column
			Type: string
		timeHeader: label/header of time column
			Type: string
	return:
		value with a type of float
	'''
	v0, t0 = pdf0.iloc[-1][itemHeader], pdf0.iloc[-1][timeHeader]
	v1, t1 = pdf1.iloc[0][itemHeader], pdf1.iloc[0][timeHeader]
	t =  datetime.combine(t1.date(), datetime.min.time())
	dt0 = (t - t0).total_seconds()/3600
	dt1 = (t1 - t).total_seconds()/3600
	v = ( v0 * dt1 + v1 * dt0 ) / ( dt0 + dt1 )
	return v

def addFirst(pdf0, pdf1, itemHeader, timeHeader):
	'''Add first row at 0:00. When length of pdf0 greater than 1, interpolate it. Or, use first row of pdf1.
	
	Parameters:
		pdf0: data befor 0:00
			Type: pandas.DataFrame object
		pdf1: data after 0:00
			Type: pandas.DataFrame object
		itemHeader: label/header of variable column
			Type: string
		timeHeader: label/header of time column
			Type: string
	return:
		pandas.DataFrame object
	'''
	first = pdf1.iloc[:1].copy().reset_index(drop=True)
	first.loc[0, timeHeader] = datetime.combine(first.loc[0, timeHeader].date(), datetime.min.time())
	if len(pdf0) > 1:
		first.loc[0, itemHeader] = interpolateByTime(pdf0, pdf1, itemHeader, timeHeader)
	pdf = pd.concat([pdf1, first]).sort_values(by=timeHeader).reset_index(drop=True)
	return pdf

def addLast(pdf0, pdf1, itemHeader, timeHeader):
	'''Add last row at 0:00. When length of pdf1 greater than 1, interpolate it. Or, use last row of pdf0.
	
	Parameters:
		pdf0: data befor 0:00
			Type: pandas.DataFrame object
		pdf1: data after 0:00
			Type: pandas.DataFrame object
		itemHeader: label/header of variable column
			Type: string
		timeHeader: label/header of time column
			Type: string
	return:
		pandas.DataFrame object
	'''
	last = pdf0.iloc[-1:].copy().reset_index(drop=True)
	last.loc[0, timeHeader] = datetime.combine(last.loc[0, timeHeader].date(), datetime.min.time()) +  timedelta(days=1)
	if len(pdf1) > 1:
		last.loc[0, itemHeader] = interpolateByTime(pdf0, pdf1, itemHeader, timeHeader)
	pdf = pd.concat([pdf0, last]).sort_values(by=timeHeader).reset_index(drop=True)
	return pdf
	
def weightsCalculator(time):
	'''Calculate weights of time for variable(water level or flow). Actuall, they have a unit of hour.
	
	Parameters:
		time: time corresponding to the variable data
			Type: list of datetime or numpy.array of datetime
	return:
		numpy.array in float/int with a unit of hour
	'''
	dt = np.array(time[1:]) - np.array(time[:-1])
	dt = [i/np.timedelta64(1, 'h') for i in dt]	# convert timedelta object to number of hours
	weights = np.array([0] + dt) + np.array(dt + [0])
	assert weights.sum() == 48
	return weights

def weightedAverage(data, weights = None):
	'''Calculate weighted average.
	
	Parameters:
		data: variable data (water level or flow)
			Type: numpy.array like object
		weights:
	return:
		float 
	'''
	return np.average(data, weights=weights)

def dailyAverageCore(pdf, itemHeader, timeHeader, date):
	'''Calculate daily average of variable (water level/flow) weighted for a specific date
	
	Parameters:
		pdf: source data to use
			Type: pandas.DataFrame object
		itemHeader: label(s)/header(s) of variable column
			Type: string or list of string
		timeHeader: label/header of time column
			Type: string
		date: specific date
			Type: datetime.date object
		
	return:
		a record of dictionary 
	'''

	d1 = datetime.combine(date, datetime.min.time())	# 0:00 of the current day (d1)
	d0 = d1 -  timedelta(days=1)	# 0:00 of the previous day (d0)
	d2 = d1 +  timedelta(days=1)	# 0:00 of the next day (d2)
	d3 = d1 +  timedelta(days=2)	# 0:00 of the next two day (d3)
	
	record = { 'year': d1.year, 'month': d1.month, 'day': d1.day }
	
	for item in itemHeader:
		# initialize
		record[item] = np.nan
		
		subdata = filterByTime(pdf, d1, d2, timeHeader).dropna(subset=item).sort_values(by=timeHeader).reset_index(drop=True)
		
		# case 0: do nothing
		if len(subdata) == 0: continue
		
		# case 1: use it as the daily average if only one gauged data
		if len(subdata) == 1: 
			record[item] = round(subdata[item][0],2)
		else:
			# first data
			if subdata.iloc[0][timeHeader].to_pydatetime() != d1:
				subdata1 = filterByTime(pdf, d0, d1, timeHeader).dropna(subset=item).sort_values(by=timeHeader).reset_index(drop=True)
				subdata = addFirst(subdata1, subdata, item, timeHeader)
			
			# last data
			if subdata.iloc[-1][timeHeader].to_pydatetime() != d2:
				subdata2 = filterByTime(pdf, d2, d3, timeHeader).dropna(subset=item).sort_values(by=timeHeader).reset_index(drop=True)
				subdata = addLast(subdata, subdata2, item, timeHeader)
			
			# weights
			weights = weightsCalculator(subdata[timeHeader].to_numpy())
			
			# daily average
			va = weightedAverage(subdata[item].to_numpy(), weights=weights)
			record[item] = round(va,2)
	
	return record
	
def dailyAverage(pdf, itemHeader, timeHeader, parallel=False, poolSize=4):
	'''Calculate daily average of variable (water level/flow) weighted along time
	
	Parameters:
		pdf: source data to use
			Type: pandas.DataFrame object
		itemHeader: label(s)/header(s) of variable column
			Type: string or list of string
		timeHeader: label/header of time column
			Type: string
		parallel: 
			Type: boolean
		poolSize: processes
			Type: int
		
	return:
		list of dictionaries
	'''
	# start date and enddate
	date0 = pdf[timeHeader].min().date()
	date1 = pdf[timeHeader].max().date()
	
	# parallel
	if parallel:
		pool = ThreadPool(poolSize)
		records = pool.starmap(dailyAverageCore, zip(repeat(pdf), repeat(itemHeader), repeat(timeHeader), daterange(date0, date1)))
		pool.close()
		pool.join()
	else:
		records = []
		while date0 < date1:
			record = dailyAverageCore(pdf, itemHeader, timeHeader, date0)
			records.append(record)
			# next day
			date0 = date0 +  timedelta(1)
	
	return records