# Time weighted average for water

[![PyPI version](https://badge.fury.io/py/time-weighted-average-for-water.svg)](https://badge.fury.io/py/time-weighted-average-for-water)
![PyPI - Downloads](https://img.shields.io/pypi/dm/time-weighted-average-for-water)
[![Downloads](https://pepy.tech/badge/time-weighted-average-for-water)](https://pepy.tech/project/time-weighted-average-for-water)

It is a tool to calculate time weighted average for gauged water level, flow, or something similar.

## Table of contents
- [Installation, update and uninstallation](#installation--update-and-uninstallation)
  * [To install](#to-install)
  * [To update](#to-update)
  * [To uninstall](#to-uninstall)
- [Usage](#usage)
- [How to cite?](#how-to-cite)
- [Changelog](#changelog)

## Installation, update and uninstallation

### To install

Quick installation with `pip`:
```bash
pip install time-weighted-average-for-water
```
Or from github:
```bash
pip install git+https://github.com/longavailable/time-weighted-average-for-water
```
Also, you can just copy related functions from *[twaw/twaw.py]* to your work.

### To update

```bash
pip install --upgrade time-weighted-average-for-water
```

### To uninstall

```bash
pip uninstall time-weighted-average-for-water
```

## Usage

See *[tests/001-daily-average.py]*.
```python
import pandas as pd
from twaw import dailyAverage

# load data
url_demodata = 'https://raw.githubusercontent.com/longavailable/datarepo02/main/data/twaw/test-data-for-twaw.csv'
data = pd.read_csv(url_demodata)
data['time'] = pd.to_datetime(data['time'])

# usages
results1 = dailyAverage(data, itemHeader=['Z', 'Q'], timeHeader='time')
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
```

## How to cite

If this tool is useful to your research, 
<a class="github-button" href="https://github.com/longavailable/time-weighted-average-for-water" aria-label="Star longavailable/time-weighted-average-for-water on GitHub">star</a> and cite it as below:
```
Xiaolong Liu. (2023, March 02). longavailable/time-weighted-average-for-water. Zenodo. 
```

## Changelog

### v0.1.0

- First release.

[Time weighted average for water]: https://github.com/longavailable/time-weighted-average-for-water
[twaw/twaw.py]: https://github.com/longavailable/time-weighted-average-for-water/blob/main/twaw/twaw.py
[tests/001-daily-average.py]: https://github.com/longavailable/time-weighted-average-for-water/blob/main/tests/001-daily-average.py

