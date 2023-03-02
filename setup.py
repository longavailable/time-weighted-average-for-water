
__version__ = '0.1.0'

from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
	long_description = fh.read()

setup(
	name='time-weighted-average-for-water',
	version=__version__,
	author='Xiaolong "Bruce" Liu',
	author_email='liuxiaolong125@gmail.com',
	description='A tool to calculate time weighted average for water level / flow.',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/longavailable/time-weighted-average-for-water',
	packages=find_packages(),
	classifiers=[
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
	],
	python_requires='>=3.6',
	install_requires=[
		'numpy',
		'pandas',
	 ],
)