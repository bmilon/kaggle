# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:22:46 2019

@author: mbhattac
"""


from pyspark.sql import SparkSession
from pyspark.sql.functions import lit,rand 

spark=SparkSession.builder.appName('data_processing').getOrCreate()


import time
start = time.time()


train_identity = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('C:\\Users\\mbhattac\\OneDrive - HERE Global B.V-\\books\\coding\\fraud detection KNIME\\ieee-fraud-detection\\train_identity.csv')
train_transaction = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('C:\\Users\\mbhattac\\OneDrive - HERE Global B.V-\\books\\coding\\fraud detection KNIME\\ieee-fraud-detection\\train_transaction.csv')


train_identity.alias('ti')
train_transaction.alias('tt')


intersection = set(train_identity.columns).intersection(set(train_transaction.columns))

newcols = set(train_identity.columns).difference(intersection)
#train =  train_transaction.withColumn(newcols,rand(len(newcols)))
train  = train_transaction.join(train_identity,train_identity.TransactionID  == train_transaction.TransactionID,how='left')
train.collect()


for cols in newcols:
    pass;
    #train[cols].flatMapValues
#train_identity, train_transaction


print('execution time : ',time.time() - start)

#for column in train.columns:
#    print(train.select(column).distinct().count())
#train_transaction.join()