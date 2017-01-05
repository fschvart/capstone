# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 00:58:00 2016

@author: Fabian Schvartzman
"""

#all reuqired imports
from datetime import datetime as dt 
from datetime import date, timedelta
import urllib
import pandas as pd
from bs4 import BeautifulSoup
from yahoo_finance import Share
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import BernoulliRBM as rbm
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

# Get the list of S&P 500 symbols
def scrape_sp500():
    page = urllib.request.urlopen("http://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(page,"lxml")

    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = list()
    for row in table.findAll('tr'):
        col = row.findAll('td')
        if len(col) > 0:
            ticker = str(col[0].string.strip())
            tickers.append(ticker)
    return tickers

# Hit raio calculation helper
def hit_ratio(pred,test,prev) :
    if pred - prev > 0 and test - prev > 0 :
        return 1
    elif pred - prev < 0 and test - prev < 0 :
        return 1
    else :
        return 0
    
timenow = date
# Get the current list of S&P 500 shares
sp = scrape_sp500()

# Our time window
delta = timedelta(days = 60)

# Hit lists for hit ratio calculations
hit_ridge = list()
hit_rbm = list()
hit_mlp = list()

# Repeat this process for each of the S&P 500 Shares
for stock in sp :
    share_info = Share(stock).get_historical(str(timenow.today() - 1.2*delta) ,timenow.today().isoformat())
    # Build's the dates list and the train labels
    dates = list()
    train_labels = pd.DataFrame(columns = ['Test'])
    t = 1
    for i in share_info :
        if dt.strptime(i['Date'], "%Y-%m-%d").date() >= timenow.today() - delta :
            dates.append(i['Date'])        
            train_labels.set_value(t, 'Test', i['Close'])
            t+=1


# Build the training and test features
    train_features = pd.DataFrame( index=dates[1:], columns = range(1,26))
    test_features = pd.DataFrame( index=dates[1:2], columns = range(1,26))
    for i in range(0,len(dates)) :
        l=1
        for j in share_info :
            if l == 25 :
                break           
            elif dt.strptime(j['Date'], "%Y-%m-%d").date() >= dt.strptime(dates[i], "%Y-%m-%d").date()  - delta/6 and dt.strptime(j['Date'], "%Y-%m-%d").date() <= dt.strptime(dates[i], "%Y-%m-%d").date() and l<= 25:
                train_features.set_value(dates[i],l, j['Open'])
                l+=1
                train_features.set_value(dates[i],l, j['Close'])
                l+=1
                train_features.set_value(dates[i],l, j['High'])
                l+=1
                train_features.set_value(dates[i],l, j['Low'])
                l+=1
                train_features.set_value(dates[i],l, j['Volume'])
                l+=1

# separate the test and training data
    test_label = train_labels.iloc[0]
    train_labels.drop([1], inplace=True)
    test_features.loc[train_features.index[0]] = train_features.iloc[0]
    train_features.drop(dates[1], inplace=True)


# Kernel Ridge model
    rid = KernelRidge(alpha = 2.0 , kernel='linear')
    rid.fit(train_features,train_labels)
    pred = rid.predict(test_features)
    hit_ridge.append(hit_ratio(pred[0][0],float(test_label.values[0]),float(train_labels.iloc[0].values[0])))

#Neural Networks

# Bernoulli's Restricted Boltzmann Machines
    fitrbm = rbm(learning_rate = 0.001, n_iter = 1500, n_components = 15)
    rbmrid = KernelRidge( alpha = 2.5, kernel='linear')
    Classifier = Pipeline(steps = [('rbm', fitrbm), ('Ridge',rbmrid)])
    Classifier.fit(train_features,train_labels)
    predrbm = Classifier.predict(test_features)
# Add this point to the hit array
    hit_rbm.append(hit_ratio(predrbm[0],float(test_label.values[0]),float(train_labels.iloc[0].values[0])))
# Multi-Layer Preceptron Regressor
    mlpfit = MLPRegressor(max_iter = 5500, activation='logistic')
    mlpfit.fit(train_features, train_labels.values.ravel())
    predmlp = mlpfit.predict(test_features)
# Add this point to the hit array
    hit_mlp.append((hit_ratio(predmlp[0],float(test_label.values[0]),float(train_labels.iloc[0].values[0]))))
    

# Sum the hits in the hit lists
su_rid = 0
su_rbm = 0
su_mlp = 0

# Go over the entire hit array and sum the hits
for i in range(len(hit_mlp)) :
    if hit_ridge[i] == 1:
        su_rid+=1
    if hit_rbm[i] == 1:
        su_rbm+=1
    if hit_mlp[i] == 1 :
        su_mlp+=1
        
# Print results
print ('Kernel Ridge Regression hit ratio: ',su_rid/len(hit_ridge))
print ('RBM hit ratio: ',su_rbm/len(hit_rbm))
print ('MLP hit ratio: ',su_mlp/len(hit_mlp))
