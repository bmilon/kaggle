import pandas as  pd 
import numpy as np 
from statistics import *
import seaborn as sns 
from math import *
from datetime import datetime
startTime = datetime.now()
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix,explained_variance_score
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

base='C:\\Users\\mbhattac\\Downloads\\ashrae\\ashrae-energy-prediction\\'

# remove the zeros data  
train  = pd.read_csv(base + 'train.csv',parse_dates=['timestamp'])
zero_indexes=train['meter_reading']  ==  0
newtrain=  train[~zero_indexes]
train_zeros=train[zero_indexes]
train=pd.concat([newtrain,train_zeros.sample(frac=0.1)],sort=True)

train['building_id']=train['building_id'].astype('category')
train['meter']=train['meter'].astype('category')

building_metadata  = pd.read_csv(base + 'building_metadata.csv')
building_metadata['floor_count']=building_metadata['floor_count'].fillna(median(building_metadata['floor_count'].dropna())).astype(int)
building_metadata['year_built']=building_metadata['year_built'].fillna(median(building_metadata['year_built'].dropna()))
building_metadata['site_id']=building_metadata['site_id'].astype('category')
building_metadata['building_id']=building_metadata['building_id'].astype('category')
building_metadata['year_built']=building_metadata['year_built'].astype(int)

label_encoder = LabelEncoder()
building_metadata['primary_use'] = label_encoder.fit(building_metadata['primary_use']).transform(building_metadata['primary_use'])

train_weather = pd.read_csv(base + 'weather_train.csv',parse_dates=['timestamp'])
train_weather['site_id']=train_weather['site_id'].astype('category')

for items in train_weather.select_dtypes(include=['float64']).columns:
    train_weather[items] = train_weather[items].fillna(median(train_weather[items].dropna()))

trainnew = train.join(building_metadata.set_index('building_id'),on='building_id',how='inner')
trainnewnew = trainnew.merge(train_weather,how='inner',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

finaltraindata=trainnewnew 

y_train = finaltraindata['meter_reading']
x_train =  finaltraindata.drop(columns=['meter_reading','timestamp','site_id','building_id'])

# split in train and test

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.01)



regressor_xgb = XGBRegressor(max_depth=10,
                min_child_weight=10,
                subsample=0.5,
                verbosity=3,
                colsample_bytree=0.6,
                #objective=custom_metric,
                n_estimators=100,
                learning_rate=0.5)


#regressor_xgb.fit(x_train,y_train)
#
## print validation score 
#
#moo  = regressor_xgb.predict(x_valid)


regressor_GBM = GradientBoostingRegressor(n_estimators=10,learning_rate=0.5,max_depth=10,alpha=0.95)


regressor_GBM.fit(x_train,y_train)

# print validation score 

moo  = regressor_GBM.predict(x_valid)

print('validation score is ',explained_variance_score(y_valid, moo))




# begining testing data 
test  = pd.read_csv(base + 'test.csv',parse_dates=['timestamp'])
test['meter']=test['meter'].astype('category')
test['building_id']=test['building_id'].astype('category')

test_weather = pd.read_csv(base + 'weather_test.csv',parse_dates=['timestamp'])

for items in test_weather.select_dtypes(include=['float64']).columns:
    test_weather[items] = test_weather[items].fillna(median(test_weather[items].dropna()))

testnew = test.join(building_metadata.set_index('building_id'),on='building_id',how='inner')
testnewnew = testnew.merge(test_weather,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

mask=testnewnew['wind_direction'].isnull() 
testdatawithnoweather = testnewnew.loc[mask]
testdatawithweather = testnewnew.loc[~mask]

summary_test__weather = test_weather.groupby('site_id').median()
summary_test__weather['cloud_coverage']=np.zeros(summary_test__weather.wind_direction.shape)

for sites in set(test_weather['site_id']):
    summary_test__weather['cloud_coverage'][sites]=median(test_weather['cloud_coverage'].filter(items=[sites]))


testdatawithnoweather=testdatawithnoweather[testnew.columns]
testdatawithnoweather  = testdatawithnoweather.join(summary_test__weather,on='site_id')

finaltestdata=pd.concat([testdatawithweather,testdatawithnoweather],sort=True)


x_test =  finaltestdata.drop(columns=['timestamp','site_id','building_id','row_id'])
#
predicted_values_xgboost=regressor_GBM.predict(x_test)

print('Execution time was : ',datetime.now() - startTime)

foo = pd.DataFrame(predicted_values_xgboost ,columns=['meter_reading'])
foo['row_id']=foo.index
foo.to_csv(base + 'submission_ashrae_py.csv',index=False )

#predicted_values = train_scaler.inverse_transform(predicted_values_xgboost)
#predicted_values = pd.DataFrame([predicted_values_DT,predicted_values_ada,predicted_values_GBM,predicted_values_cat]).mean()
#foo = pd.DataFrame(predicted_values_xgboost ,columns=['meter_reading'])
#foo['row_id']=foo.index
#foo.to_csv(base + 'submission_ashrae_py.csv',index=False )


#################################end of code 
#X_train =  finaltestdata.drop(columns=['meter_reading','timestamp','site_id','building_id'])
#y_predict = regressor.predict(X)

#predicted_values_DT = regressor_DT.predict(X_test)
#predicted_values_ada = regressor_ada.predict(x_test)
#predicted_values_GBM = regressor_GBM.predict(X_test)
#predicted_values_cat = regressor_cat.predict(X_test)
#predicted_values_LGBM = gbm.predict(x_test, num_iteration=gbm.best_iteration)

#print(confusion_matrix(y, y_predict))
#print(classification_report(int(y), int(y_predict)))
#items=300000
#sns.scatterplot(y[1:items],regressor.predict(X_train[1:items]))


#foo = testdatawithnoweather.columns.drop(['wind_direction','sea_level_pressure','dew_temperature','air_temperature','precip_depth_1_hr','wind_speed','cloud_coverage',])
#testdatawithnoweather = testdatawithnoweather[foo]

#newcols=[];
#for cols in  testdatawithnoweather.columns:
#    if (cols in  ['site_id','timestamp'])  | ( cols not in test_weather.columns):
#        newcols.append(cols)

#
        


#lgb_train = lgb.Dataset(x_train, y_train)
#lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)
#params = {
#    'boosting_type': 'gbdt',
#    'objective': 'regression',
#    'metric': {'l2', 'l1'},
#    'num_leaves': 31,
#    'learning_rate': 0.05,
#    'feature_fraction': 0.9,
#    'bagging_fraction': 0.8,
#    'bagging_freq': 5,
#    'verbose': 1
#}
#
#gbm = lgb.train(params,
#                lgb_train,
#                num_boost_round=20,
#                valid_sets=lgb_eval,
#                early_stopping_rounds=5)


# array of predictors begin 
#regressor_ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8),n_estimators=10, random_state=np.random.RandomState(1))
#regressor_ada.fit(x_train, y_train)



#regressor_DT = DecisionTreeRegressor(random_state = 0)
#regressor_DT.fit(x_train, y_train)
#
#regressor_cat = CatBoostRegressor(iterations=500, 
#                          depth=8, 
#                          verbose=True,
#                          learning_rate=0.5, 
#                          loss_function='RMSE')
#
#regressor_cat.fit(x_train,y=y_train)
#
#
#regressor_GBM = GradientBoostingRegressor(n_estimators=100,learning_rate=0.5)
#regressor_GBM.fit(x_train,y_train)
# array of predictors close 


#reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
#reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
#reg3 = LinearRegression()
#ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
#ereg = ereg.fit(X, y)

#def custom_metric(target,predict):
#    return sqrt(mean(predict**2 - target**2))

#import pandas as  pd 
#import numpy as np 
#from statistics import *
#import seaborn as sns 
#from math import *
#from datetime import datetime
#startTime = datetime.now()
#from sklearn import preprocessing
#
#from sklearn.tree import DecisionTreeRegressor
#from catboost import Pool, CatBoostRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import GradientBoostingRegressor
#from xgboost import XGBRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import VotingRegressor
#
#import matplotlib.pyplot as plt
#from sklearn.tree import export_graphviz  
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#import lightgbm  as lgb
#import imblearn
#
#from sklearn.metrics import explained_variance_score
#from sklearn import preprocessing
#"""
#useimputer for missing values 
#test voting regressor 
#use different cross validations 
#increase sampling data rate while training regression tree 
#change to make in the code
#train zero meter reading examples seperately 
#split data for zeros meter reading and other
#train a seperate data modelfor entries in trainining data without weather data 
#"""
#
#
#base='C:\\Users\\mbhattac\\Downloads\\ashrae\\ashrae-energy-prediction\\'
#
## trainind data 
#train  = pd.read_csv(base + 'train.csv',parse_dates=['timestamp'])
#building_metadata  = pd.read_csv(base + 'building_metadata.csv')
#train_weather = pd.read_csv(base + 'weather_train.csv',parse_dates=['timestamp'])
#
#
##train_scaler  = preprocessing.StandardScaler().fit(train.select_dtypes(include=['float64']))
##train['meter_reading'] = train_scaler.transform(train.select_dtypes(include=['float64']))
#train['meter']=train['meter'].astype('category')
#train['building_id']=train['building_id'].astype('category')
#
#
#
##building data 
#building_metadata['primary_use']=building_metadata['primary_use'].astype('category')
#building_metadata['floor_count']=building_metadata['floor_count'].fillna(mode(building_metadata['floor_count'].dropna())).astype('category')
#building_metadata['site_id']=building_metadata['site_id'].astype('category')
#building_metadata['building_id']=building_metadata['building_id'].astype('category')
#for items in ['square_feet','year_built']:
#    building_metadata[items] = building_metadata[items].fillna(building_metadata[items].dropna().mode())
#
#
## train weather data 
#train_weather['cloud_coverage']=train_weather['cloud_coverage'].fillna(mode(train_weather['cloud_coverage'].dropna()))
#for items in ['air_temperature', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure','wind_direction', 'wind_speed']:
#    train_weather[items] = train_weather[items].fillna(mode(train_weather[items].dropna()))
#
##train_weather['precip_depth_1_hr']= train_weather['precip_depth_1_hr']*1000
#
#
#trainnew = train.join(building_metadata.set_index('building_id'),on='building_id')
#trainnewnew = trainnew.merge(train_weather,how='inner',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
#
#
##mask=trainnewnew['wind_direction'].isnull() 
##traindatawithnoweather = trainnewnew.loc[mask]
##traindatawithweather = trainnewnew.loc[~mask]
##
##summary_train__weather = train_weather.groupby('site_id').mean()
##summary_train__weather['cloud_coverage']=np.zeros(summary_train__weather.wind_direction.shape)
##
##for sites in set(train_weather['site_id']):
##    summary_train__weather['cloud_coverage'][sites]=mode(train_weather['cloud_coverage'].filter(items=[sites]))
##
##
##traindatawithnoweather=traindatawithnoweather[trainnew.columns]
##traindatawithnoweather  = traindatawithnoweather.join(summary_train__weather,on='site_id')
##
##finaltraindata=pd.concat([traindatawithweather,traindatawithnoweather],sort=True)
#
#finaltraindata=trainnewnew.loc[ trainnewnew['meter_reading'] != 0 ]
#finaltraindata = finaltraindata.reindex(sorted(finaltraindata.columns), axis=1)
#
#le_primary_use = preprocessing.LabelEncoder()
#le_primary_use.fit(finaltraindata['primary_use'])
#finaltraindata['primary_use']=le_primary_use.transform(finaltraindata['primary_use'])
#
#
#le_floor_count = preprocessing.LabelEncoder()
#le_floor_count.fit(finaltraindata['floor_count'])
#finaltraindata['floor_count']=le_floor_count.transform(finaltraindata['floor_count'])
#
#le_meter = preprocessing.LabelEncoder()
#le_meter.fit(finaltraindata['meter'])
#finaltraindata['meter']=le_meter.transform(finaltraindata['meter'])
#
#
#y_train = finaltraindata['meter_reading']
#x_train =  finaltraindata.drop(columns=['meter_reading','timestamp','site_id','building_id'])
#
## split in train and test
#
#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.20)
#
#
#smote = imblearn.over_sampling.SMOTE(ratio='minority')
##X_sm, y_sm = smote.fit_sample(x_train, y_train)
##x_train['floor_count_new'] = x_train['floor_count'].apply(lambda x: int(x))
#
#
#
#regressor_xgb = XGBRegressor(max_depth=100,
#                min_child_weight=10,
#                subsample=0.5,
#                verbosity=3,
#                colsample_bytree=0.6,
#                #objective=custom_metric,
#                n_estimators=100,
#                learning_rate=0.5)
#
#
#regressor_xgb.fit(x_train,y_train)
#
## print validation score 
#
#moo  = regressor_xgb.predict(x_valid)
#
#print('validation score is ',explained_variance_score(y_valid, moo))
#
## begining testing data 
#
#
#
## test weather data 
#test_weather = pd.read_csv(base + 'weather_test.csv',parse_dates=['timestamp'])
#test_weather['cloud_coverage']=test_weather['cloud_coverage'].fillna(mode(test_weather['cloud_coverage'].dropna()))
#for items in ['air_temperature', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure','wind_direction', 'wind_speed']:
#    test_weather[items] = test_weather[items].fillna(mode(test_weather[items].dropna()))
#
#
#testnew = test.join(building_metadata.set_index('building_id'),on='building_id')
#testnewnew = testnew.merge(test_weather,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
#
#
#mask=testnewnew['wind_direction'].isnull() 
#testdatawithnoweather = testnewnew.loc[mask]
#testdatawithweather = testnewnew.loc[~mask]
#
#summary_test__weather = test_weather.groupby('site_id').mean()
#summary_test__weather['cloud_coverage']=np.zeros(summary_test__weather.wind_direction.shape)
#
#for sites in set(test_weather['site_id']):
#    summary_test__weather['cloud_coverage'][sites]=mode(test_weather['cloud_coverage'].filter(items=[sites]))
#
#
#testdatawithnoweather=testdatawithnoweather[testnew.columns]
#testdatawithnoweather  = testdatawithnoweather.join(summary_test__weather,on='site_id')
#
#finaltestdata=pd.concat([testdatawithweather,testdatawithnoweather],sort=True)
#
#finaltestdata['primary_use']=le_primary_use.transform(finaltestdata['primary_use'])
#finaltestdata['floor_count']=le_floor_count.transform(finaltestdata['floor_count'])
#finaltestdata['meter']=le_meter.transform(finaltestdata['meter'])
#
#
#x_test =  finaltestdata.drop(columns=['timestamp','site_id','building_id','row_id'])
#
#predicted_values_xgboost=regressor_xgb.predict(x_test)
#
#print('Execution time was : ',datetime.now() - startTime)
#
##predicted_values = train_scaler.inverse_transform(predicted_values_xgboost)
##predicted_values = pd.DataFrame([predicted_values_DT,predicted_values_ada,predicted_values_GBM,predicted_values_cat]).mean()
#foo = pd.DataFrame(predicted_values_xgboost ,columns=['meter_reading'])
#foo['row_id']=foo.index
#foo.to_csv(base + 'submission_ashrae_py.csv',index=False )
#
