

import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd


def process(path):
	data = pd.read_csv(path,usecols=["Species","Weight","Length1","Length2","Length3","Height","Width"])
	print(data)



	names=list(data.columns)
	


	fig, ax = plt.subplots(figsize=(15,7))    	
	ncols=3
	plt.clf()
	f = plt.figure(1)
	f.suptitle(" Data Histograms", fontsize=12)
	vlist = list(data.columns)
	nrows = len(vlist) // ncols
	if len(vlist) % ncols > 0:
		nrows += 1
	for i, var in enumerate(vlist):
		plt.subplot(nrows, ncols, i+1)
		plt.hist(data[var].values, bins=15)
		plt.title(var, fontsize=10)
		plt.tick_params(labelbottom='off', labelleft='off')
	plt.tight_layout()
	plt.subplots_adjust(top=0.88)
	plt.savefig('static/results/DataHistograms.png')



	fig, ax = plt.subplots(figsize=(15,7))
	data['Species'].value_counts().plot.bar(rot=0)
	n=data['Species'].value_counts().plot.bar(rot=0)
	ax.title.set_text('Number of Records Species')
	ax.set_ylabel('Sum Value')
	plt.savefig('static/results/Number of Records Species.png')



	fig, ax = plt.subplots(figsize=(15,7))
	data[["Species", "Weight"]].groupby("Species").sum().plot.bar(stacked=True,ax=ax)
	ax.title.set_text('Total Species Weight')
	ax.set_ylabel('Sum Value')
	plt.savefig('static/results/Total Species Weight.png')

	a = data[["Species", "Weight"]].groupby("Species").sum()
	print(a)
	a.plot.pie(subplots=True,figsize=(20, 20))
	plt.title('Total Species Weight')

	plt.savefig('static/results/Pie-Species Weight.png')




#process("dataset.csv")	