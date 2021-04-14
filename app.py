from flask_socketio import SocketIO, send, join_room
from flask import Flask, flash, redirect, render_template, request, session, abort,url_for
import os
#import StockPrice as SP
import re
import sqlite3
import pandas as pd
import numpy as np
import requests

import Preprocess as pr
import DTALG as dt
import RFALG as rf
from flask_table import Table, Col

    
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
	return render_template('main.html')





@app.route('/main',methods=['POST'])
def main_page():
    	path=request.form['datasetfile']
    	print(path)
    	return render_template('main.html')

@app.route('/preprocess',methods=['POST'])
def preprocess():
    	path=request.form['datasetfile']
    	print(path)
    	pr.process(path)
    	return render_template('main.html',result=0)
   
@app.route('/DT',methods=['POST'])
def DT():
    	path=request.form['datasetfile']
    	print(path)
    	y_test,y_pred=dt.process(path)
    	res=[]
    	for i in range(0, len(y_test)):
    		res.append([y_test[i],y_pred[i]])
    	return render_template('main.html',result=1,res=res)

@app.route('/RF',methods=['POST'])
def RF():
    	path=request.form['datasetfile']
    	print(path)
    	y_test,y_pred=rf.process(path)
    	res=[]
    	for i in range(0, len(y_test)):
    		res.append([y_test[i],y_pred[i]])
    	return render_template('main.html',result=2,res=res)

# /////////socket io config ///////////////
#when message is recieved from the client    
@socketio.on('message')
def handleMessage(msg):
    print("Message recieved: " + msg)
 
# socket-io error handling
@socketio.on_error()        # Handles the default namespace
def error_handler(e):
    pass


  
  
if __name__ == '__main__':
    socketio.run(app,debug=True,host='127.0.0.1', port=4000)
