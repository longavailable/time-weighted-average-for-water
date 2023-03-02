# -*- coding: utf-8 -*-
'''
* Updated on 2023/03/02
* python3
**
* Test for dailyAverage
'''

import pandas as pd
from twaw import dailyAverage

# load data
url_demodata = 'https://raw.githubusercontent.com/longavailable/datarepo02/main/data/twaw/test-data-for-twaw.csv'
data = pd.read_csv(url_demodata)
data['time'] = pd.to_datetime(data['time'])

# usages
items = ['Z', 'Q']
results1 = dailyAverage(data, itemHeader=items, timeHeader='time')
results2 = dailyAverage(data, itemHeader=['Z'], timeHeader='time')
results3 = dailyAverage(data, itemHeader='Q', timeHeader='time')

# export
newdata = pd.DataFrame(data=results1)
newdata2 = newdata.dropna(subset=items, how='all').sort_values(by=['year', 'month', 'day'])
if len(newdata2) > 0:
	filename = 'test-o.csv'
	newdata2.to_csv(filename, index=False)
else:
	print('No data to export!')
