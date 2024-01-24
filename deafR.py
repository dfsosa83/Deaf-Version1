from scipy import stats
import joblib
from sklearn.linear_model import LinearRegression

#for google drive
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#gauth = GoogleAuth()
#gauth.LocalWebserverAuth()  # Opens a local web server for authentication
#drive = GoogleDrive(gauth)

#file_id = "1ndTa_rwxSZY-ssVz7Ew9htuYSTsfCsty"
#file = drive.CreateFile({'id': file_id})
#file.GetContentFile('Merged_Data.csv', mimetype='text/csv')

##import libraries
from flask import Flask #Mono
from flask import request #Mono

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

import lightgbm as lgb
import warnings
import pandas as pd
import numpy as np
import ta
import time
import pickle
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning)

#load data
origin_data = pd.read_csv('C:/Users/david/OneDrive/Documents/deaf_reload/data/Merged_Data.csv',
       low_memory=False,delimiter=',',usecols=['datetime','topWick','body','bottomWick','ticks','spread','open','high','low','close'],nrows=10000)

#origin_data = pd.read_csv('C:/Users/david/OneDrive/Documents/deaf_reload/proof_lags_1.csv',low_memory=False,delimiter=',')

# prompt: rename datetime column by timestamp_column
origin_data.rename(columns={'datetime': 'timestamp_column'}, inplace=True)

#convert datatime to_datetime
origin_data['timestamp_column'] = pd.to_datetime(origin_data['timestamp_column'])
#print(origin_data)

#drop nan if is any
df_nn = origin_data.dropna()

#create variable of data name to only chance one time
data_variable = df_nn.copy()

##add lags
one_min = 60
two_min = 120
three_min = 180
four_min = 240
five_min = 300
ten_min = 600
fifteen_min =900
twenty_min = 1200
twentyfive_min = 1500
thirty_min = 1800
fourty_min = 2400
sixty_min = 3600

#features
data_variable['month'] = data_variable['timestamp_column'].apply(lambda x: x.month)
data_variable['day'] = data_variable['timestamp_column'].apply(lambda x: x.day)
data_variable['hour'] = data_variable['timestamp_column'].apply(lambda x: x.hour)
data_variable['body_15'] = data_variable['body'].shift(fifteen_min)
data_variable['body_40'] = data_variable['body'].shift(fourty_min)
data_variable['ticks_15'] = data_variable['ticks'].shift(fifteen_min)
data_variable['ticks_25'] = data_variable['ticks'].shift(twentyfive_min)
data_variable['ticks_15'] = data_variable['ticks'].shift(fifteen_min)
data_variable['ticks_60'] = data_variable['ticks'].shift(sixty_min)
data_variable['spread_5'] = data_variable['spread'].shift(five_min)
data_variable['spread_15'] = data_variable['spread'].shift(fifteen_min)
data_variable['spread_25'] = data_variable['spread'].shift(twentyfive_min)
data_variable['spread_40'] = data_variable['spread'].shift(fourty_min)
data_variable['spread_log'] = np.log(data_variable.iloc[:, 5] + 1)
data_variable['body_log'] = np.log(data_variable.iloc[:, 2] +1)
data_variable['ticks_log'] = np.log(data_variable.iloc[:, 4] +1)
data_variable['spread_sqrt'] = np.sqrt(data_variable.iloc[:, 5])
data_variable['body_sqrt'] = np.sqrt(data_variable.iloc[:, 2])
data_variable['spread_x_body'] = data_variable.iloc[:, 5] * data_variable.iloc[:, 2] 
data_variable['spread_x_ticks'] = data_variable.iloc[:, 5] * data_variable.iloc[:, 4] 
data_variable['body_x_ticks'] = data_variable.iloc[:, 2] * data_variable.iloc[:, 4] 
##Adding a constant to avoid zero and negative values
constant = 1
#### prompt: Power Transformation
data_variable['spread_transformed'] = data_variable.iloc[:, 5] + constant
data_variable['body_transformed'] = data_variable.iloc[:, 2] + constant
data_variable['ticks_transformed'] = data_variable.iloc[:, 4] + constant
#### Applying the Box-Cox transformation
data_variable['spread_boxcox'], lam = stats.boxcox(data_variable.iloc[:, 30])
data_variable['body_boxcox'], lam = stats.boxcox(data_variable.iloc[:, 31])
data_variable['ticks_boxcox'], lam = stats.boxcox(data_variable.iloc[:, 32])

###features
selected_feature_names = [
'topWick', 
'body', 
'bottomWick', 
'spread', 
'hour', 
'day', 
'month',
'body_15', 
'body_40', 
'ticks_15', 
'ticks_25', 
'ticks_60', 
'spread_5',
'spread_15', 
'spread_25', 
'spread_40', 
'spread_log', 
'body_log',
'ticks_log', 
'spread_sqrt', 
'body_sqrt', 
'spread_boxcox', 
'body_boxcox',
'ticks_boxcox', 
'spread_x_body', 
'spread_x_ticks', 
'body_x_ticks'
]

#drop nan
df = data_variable.dropna()
print(df.head(1))

### Load the saved models from the .pkl files
model1 = pickle.load(open("C:/Users/david/OneDrive/Documents/deaf_reload//Model/rf_model_lags_v2.sav", 'rb'))
model2 = pickle.load(open("C:/Users/david/OneDrive/Documents/deaf_reload//Model/lg_model_lags_v2.sav","rb"))
model3 = pickle.load(open("C:/Users/david/OneDrive/Documents/deaf_reload//Model/hist_model_lags_v2.sav","rb"))
meta_model = pickle.load(open("C:/Users/david/OneDrive/Documents/deaf_reload//Model/meta_model_lags_v2.sav","rb"))
print("Models loaded successfully!")
##

###predictions
X_new_0 = df[selected_feature_names]

###pick last row
#X_new = X_new_0.iloc[:1]
X_new = X_new_0.tail(1)
#X_new = X_new_0.copy()
#print(X_new)
##
###predict
pred_model1 = model1.predict(X_new)
pred_model2 = model2.predict(X_new)
pred_model3 = model3.predict(X_new)
##
### Combine the predictions into a single array
base_model_preds = [pred_model1, pred_model2, pred_model3]
base_model_preds = np.array(base_model_preds).T
print(base_model_preds)

### Make prediction with the meta-model
pred_meta_model = meta_model.predict(base_model_preds)
print(pred_meta_model,"Predicted by meta model all decimals!!!")
#
####decimal spaces
rounded_arr = np.round(pred_meta_model, 2)

##Print the final prediction from the meta-model
print(rounded_arr,"Predicted by meta model rounded!!!")