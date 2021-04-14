import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import linear_model
#from sklearn.model_selection cross_validation
from scipy.stats import norm

from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



def process(path):
	dataset = pd.read_csv(path)
	X = dataset.iloc[:, 1:6].values
	y = dataset.iloc[:, 6].values
	y = (y / 100).astype(int) *100
	X_train, X_test, y_train, y_test = train_test_split(X, y)

	model2=RandomForestClassifier()
	model2.fit(X_train, y_train)
	y_pred = model2.predict(X_test)
	print("predicted")
	print(y_pred)
	print("test")
	print(y_test)

	result2=open("static/results/resultRF.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()
	
	mse=abs(round(mean_squared_error(y_test, y_pred),2))/1000
	mae=abs(round(mean_absolute_error(y_test, y_pred),2))
	r2=abs(round(r2_score(y_test, y_pred),2))
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR RandomForest IS %f "  % mse)
	print("MAE VALUE FOR RandomForest IS %f "  % mae)
	print("R-SQUARED VALUE FOR RandomForest IS %f "  % r2)
	rms = abs(round(np.sqrt(mean_squared_error(y_test, y_pred)),2))
	print("RMSE VALUE FOR RandomForest IS %f "  % rms)
	ac=round(accuracy_score(y_test,y_pred),2)*100
	print ("ACCURACY VALUE RandomForest IS %f" % ac)
	print("---------------------------------------------------------")
	

	result2=open('static/results/RFMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('static/results/RFMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title(' Random Forest Metrics Value')
	fig.savefig('static/results/RFMetricsValueBarChart.png') 


	group_names=['MSE', 'MAE','R2','RMSE','ACCURACY']
	group_size=acc
	subgroup_names=acc
	subgroup_size=acc
	 
	# Create colors
	a, b, c,d,e=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens,plt.cm.Oranges,plt.cm.Purples]
	 
	# First Ring (outside)
	fig, ax = plt.subplots()
	ax.axis('equal')
	mypie, _ = ax.pie(group_size, radius=1.0, labels=group_names, colors=[a(0.6), b(0.6), c(0.6),d(0.1),e(0.6)] )
	plt.setp( mypie, width=0.3, edgecolor='white')
	 
	## Second Ring (Inside)
	mypie2, _ = ax.pie(subgroup_size, radius=1.0-0.3, labels=subgroup_names, labeldistance=0.7, colors=[a(0.6), b(0.6), c(0.6),d(0.1),e(0.6)] )
	plt.setp( mypie2, width=0.4, edgecolor='white')
	plt.margins(0,0)
	 
	plt.title(' Random Forest Metrics Value')
	plt.savefig('static/results/RFMetricsValue.png')


	# set width of bar 
	barWidth = 0.25
	fig = plt.subplots(figsize =(12, 8)) 
	

	# Set position of bar on X axis 
	br1 = np.arange(len(y_pred))
	br2 = [x + barWidth for x in br1] 

	
	# Make the plot
	plt.bar(br1, y_test, color ='r', width = barWidth,edgecolor ='grey', label ='Original') 
	plt.bar(br2, y_pred, color ='g', width = barWidth,edgecolor ='grey', label ='Predicted') 
	
	
	# Adding Xticks 
	plt.xlabel('Number of Records', fontweight ='bold', fontsize = 15) 
	plt.ylabel('Fish Weight', fontweight ='bold', fontsize = 15) 
	plt.legend()
	plt.savefig('static/results/RFCompare.png')
	return y_test,y_pred

#process("dataset.csv")
	