# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:01:56 2021

@author: Rejith Reghunathan 
@email: rejithrnath@gmail.com

Reference : Python for Algorithmic Trading - By Yves Hilpisch
"""

import yfinance as yf
import schedule
from datetime import datetime, timedelta
import time
# import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold,\
  cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix,\
  accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GroupKFold
import pickle
from sklearn.metrics import recall_score

import os, pandas
import shutil
import time
import email, smtplib, ssl
import schedule
import temp.config
import requests
from bs4 import BeautifulSoup
from finta import TA

from os import system
# system("title " + "The forest")

if not os.path.exists('results'):
        os.makedirs('results')

if not os.path.exists('models'):
        os.makedirs('models')

if not os.path.exists('figure'):
        os.makedirs('figure')
        
save_path = 'results/'
filename_results = datetime.now().strftime("%Y%m%d")
completeName = os.path.join(save_path, filename_results+".txt")

    
def delete_results():
    shutil.rmtree('results')
    os.makedirs('results')
    save_path = 'results/'
    filename_results = datetime.now().strftime("%Y%m%d")
    completeName = os.path.join(save_path, filename_results+".txt") 






def random_forest(symbol, interval_time, delta_time, lags):
        
         
        #calculation of price momentum
        def MOM(df, n):
            MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
            return MOM
       
        #MACD
        def MACD(series: pd.Series,n1: int=12, n2: int=26) -> pd.Series:
          assert n1 < n2, f'n1 must be less than n2'
          return calculate_simple_moving_average(series, n1) - \
              calculate_simple_moving_average(series, n2)

        def calculate_simple_moving_average(series: pd.Series, n: int=20) -> pd.Series:
           return series.rolling(n).mean()

        #calculation of relative strength index
        def RSI(series, period):
         delta = series.diff().dropna()
         u = delta * 0
         d = u.copy()
         u[delta > 0] = delta[delta > 0]
         d[delta < 0] = -delta[delta < 0]
         u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
         u = u.drop(u.index[:(period-1)])
         d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
         d = d.drop(d.index[:(period-1)])
         rs = u.ewm(com=period-1, adjust=False).mean() / \
         d.ewm(com=period-1, adjust=False).mean()
         return 100 - 100 / (1 + rs)
        
           
        now = datetime.utcnow()
        now = now - timedelta(microseconds = now.microsecond)
        yesterday = now - timedelta(days = delta_time)
        dataset=yf.download(symbol,yesterday,now, interval=interval_time, progress = False)
       
        
        dataset[dataset.columns.values] = dataset[dataset.columns.values].ffill()
        # dataset[['Adj Close']].plot(grid=True)
        # plt.title(symbol + ' Dataset')
        # plt.show()
        dataset["returns"] = np.log(dataset['Adj Close'].shift(-1).div(dataset['Adj Close']))
        dataset[dataset.columns.values] = dataset[dataset.columns.values].ffill()
        short_window = 21 
        long_window = 55

        dataset["signal"] = np.where(dataset["returns"] > 0, 1, 0)
        dataset['MACD'] = MACD(dataset['Adj Close'])
        dataset['MOM10'] = MOM(dataset['Adj Close'], 10)
        
        
        dataset['RSI10'] = RSI(dataset['Adj Close'], 10)
        
        
        dataset["sma"] = dataset['Adj Close'].rolling(short_window).mean() - dataset['Adj Close'].rolling(long_window).mean()
        dataset["boll"] = (dataset['Adj Close'] - dataset['Adj Close'].rolling(short_window).mean()) / dataset['Adj Close'].rolling(short_window).std()
        dataset["min"] = dataset['Adj Close'].rolling(short_window).min() / dataset['Adj Close'] - 1
        dataset["max"] = dataset['Adj Close'].rolling(short_window).max() / dataset['Adj Close'] - 1
        dataset["vol"] = dataset["Volume"].rolling(3).std()
        dataset["OBV"] = TA.OBV(dataset)
        dataset['EMA_short'] = dataset['Adj Close'].ewm(span = short_window, adjust = False).mean()
        dataset['EMA_long'] = dataset['Adj Close'].ewm(span = long_window, adjust = False).mean()
        dataset['DMA_short'] = 2*dataset['EMA_short'] - (dataset['EMA_short'].ewm(span = short_window, adjust = False).mean())
        dataset['DMA_long'] = 2*dataset['EMA_long'] - (dataset['EMA_long'].ewm(span = long_window, adjust = False).mean())
        dataset['H-L']=abs(dataset['High']-dataset['Low'])
        dataset['H-PC']=abs(dataset['High']-dataset['Adj Close'].shift(1))
        dataset['L-PC']=abs(dataset['Low']-dataset['Adj Close'].shift(1))
        dataset['TR']=dataset[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
        dataset['ATR'] = dataset['TR'].ewm(span=short_window,adjust=False,min_periods=short_window).mean()
        dataset ['STOCH'] =TA.STOCH(dataset)
        dataset ['VAMA'] = TA.VAMA(dataset)
        # dataset ['OBX'] = yf.download('OBX.OL',yesterday,now, interval=interval_time, progress = False)['Close']
        #etoro technical indicators
        dataset ['RSI'] =TA.RSI(dataset,short_window)
        dataset ['STOCHRSI_14'] = TA.STOCHRSI(dataset,short_window)
        dataset['CCI_14'] = TA.CCI(dataset)
        dataset ['STOCH_14'] =TA.STOCH(dataset)
        dataset ['AO_50_20'] =TA.AO(dataset,long_window,short_window)
        dataset ['HULL'] =TA.HMA(dataset,short_window)
        dataset['adx'] = TA.ADX(dataset)
        dataset['cmo'] = TA.CMO(dataset)
        dataset['wil'] = TA.WILLIAMS(dataset)
        dataset['adl'] = TA.ADL(dataset)
      
        
        cols = []
       
        # features = ["STOCH","MACD","MOM50","MOM10", "MOMl.nnnnnnnnnnnnnnnnnn 30", "RSI10", "RSI30", "RSI50", "sma", "boll","vol", "mom_30","mom_7", "ATR"]
        #features=["OBV","max","min","vol","RSI","adx","cmo","wil","adl","STOCHRSI_14","MACD","CCI_14","STOCH_14","AO_50_20","ATR","MOM10","STOCH","RSI10","VAMA", "ATR","EMA_short","EMA_long","DMA_short","DMA_long",  "sma", "boll","HULL"]
        features = ["Open","High","Low","Close","OBV","MACD","RSI"]
        features_dataset = dataset.copy()
        for f in features:
          for lag in range(1, lags+1 ):
            col = "{}_lag_{}".format(f, lag)
            dataset[col] = features_dataset[f].shift(lag)
            #dataset[col] = np.log(features_dataset[f].shift(1).div(features_dataset[f].shift(2)))
            cols.append(col)

        present_features = ["Open"]
        for g in present_features:
          for lag in range(0, lags ):
            col = "{}_lag_{}".format(g, lag)
            dataset[col] = features_dataset[g].shift(lag)
            #dataset[col] = np.log(features_dataset[f].shift(1).div(features_dataset[f].shift(2)))
            cols.append(col)


        dataset.dropna(inplace = True)
       
        # split out validation dataset for the end
        dataset_length = dataset.shape[0]
        split = int(dataset_length * 0.75)
        subset_dataset= dataset.copy()
        Y= subset_dataset["signal"]
        # X = subset_dataset.loc[:, ~dataset.columns.isin(['signal', 'Close'])]
        X = dataset.copy()
        X_return_calc = dataset[split:]

        X_train, X_validation = X[:split], X[split:]
        Y_train, Y_validation = Y[:split], Y[split:]
         

        

        # hyper parameter tuning
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.model_selection import GridSearchCV
        # define models and parameters
        model = RandomForestClassifier()
       
        
        # define grid search
        param_grid = {
          'bootstrap': [True],
          'max_depth': [80,160,240],
          'n_estimators': [200,400,600]
          }
        
        rf = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=0)
        grid_result = rf.fit(X_train[cols], Y_train)
        # summarize results
        print("Best RF : %f using %s" % (grid_result.best_score_, grid_result.best_params_))
       

        # estimate accuracy on validation set
        predictions = rf.predict(X_validation[cols])
        print(symbol + " - Accuracy Score ->")
        print(accuracy_score(Y_validation, predictions))
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(np.sign(Y_validation), predictions)
        print(cm)
        
        
        backtestdata = pd.DataFrame(index=X_validation.index)
        backtestdata['signal_pred'] = predictions
        backtestdata['signal_actual'] = Y_validation
        
        backtestdata[['signal_pred','signal_actual']].plot()
        plt.title(symbol + ' Backtesting' + '  Acc: ->  '+ str(accuracy_score(Y_validation, predictions)))
        
        fig_path = os.path.join('figure',symbol+'_test_plot.png')
        plt.savefig(fig_path)

        f = open(completeName, "a")
        print(('Precision Recall Fscore  -> '), file=f)
        print('Precision Recall Fscore  -> ')
        from sklearn.metrics import precision_recall_fscore_support
        print(precision_recall_fscore_support(Y_validation, predictions, average='binary'))
        print(precision_recall_fscore_support(Y_validation, predictions, average='binary'), file=f)
        f.close()
      

        # print(dataset.loc[dataset.index[-1], "signal"])
        # print("Prediction ->")
        print('')
        if ( dataset.loc[dataset.index[-1], "signal"] == 1):
          f = open(completeName, "a")
          print(symbol + ' - > +++ Buy +++' +' Accuracy -> ' + str(accuracy_score(Y_validation, predictions)))
          print(symbol + " - > +++ Buy +++" +' Accuracy -> ' +str(accuracy_score(Y_validation, predictions)), file=f)
          f.close()
          
        # elif ( dataset.loc[dataset.index[-1], "signal"] == 0):
        #   print('Neutral') 
         
        else:
          f = open(completeName, "a")
          print(symbol + ' - > ---- Sell ----' +' Accuracy -> ' + str(accuracy_score(Y_validation, predictions)))
          print(symbol + " - > ---- Sell ----" + ' Accuracy -> ' +str(accuracy_score(Y_validation, predictions)), file=f)
          f.close()
          
        print('')

interval_time = '1d'
delta_time =30
lags =3



    
def email_export():
    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    
    subject = "**** updated ****** CRYPTO ML Results "+ str(datetime.now())
    body = "Email with attachment "
    
    sender_email = temp.config.sender_email
    receiver_email = temp.config.receiver_email
    password = temp.config.password

        
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = sender_email
    message["Subject"] = subject
    message["Bcc"] = ", ".join(receiver_email)  # Recommended for mass emails
    
    # Add body to email
    message.attach(MIMEText(body, "plain"))
    
    
    # Open PDF file in binary mode
    with open(completeName, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    
    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)
    
    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {completeName}",
    )
    
    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()
    
    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)
        print("Emailed!")
        delete_results()


def download_and_email():
    
    try:
      f = open(completeName, "a")
      print ("Start the Forest — CRYPTO Market -> %s \n" % time.ctime(), file=f) 
      print ("*******************************************************************" , file=f)
      print ("*******************************************************************" )
      from datetime import datetime
      now = datetime.now()
      print('')
      print("start =", now)
      print(now, file=f)
      f.close()
      print("Program : Random_Forest")
      
      # with open("/content/gdrive/MyDrive/Algotrading/Algotrading/Colab/ML/Logistic_Regression/input.csv") as f:
      with open("CRP.csv") as f:  
              lines = f.read().splitlines()
              for symbol in lines:
                
                    # print(symbol)
                    random_forest(symbol,interval_time,delta_time,lags)
                
    except:
         pass
    
    f = open(completeName, "a")
    now = datetime.now()
    print('')
    print("end =", now)
    print(now, file=f)
    print ("*******************************************************************" )
    print ("*******************************************************************" , file=f)
    f.close()
    print("End!")
    #email_export()  




def main():
    time.sleep(1) 
    print("The Forest Crypto Market!!")
    download_and_email()
    trading_time = [ "00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23"]
    for x in trading_time:
      
        schedule.every().monday.at(str(x)+":30").do(download_and_email)
        schedule.every().tuesday.at(str(x)+":30").do(download_and_email)
        schedule.every().wednesday.at(str(x)+":30").do(download_and_email)
        schedule.every().thursday.at(str(x)+":30").do(download_and_email)
        schedule.every().friday.at(str(x)+":30").do(download_and_email)
        schedule.every().saturday.at(str(x)+":30").do(download_and_email)
        schedule.every().sunday.at(str(x)+":30").do(download_and_email)
 
    while True:
        schedule.run_pending()
        time.sleep(10)                    



if __name__ == "__main__":
    main()

