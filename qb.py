

 

import matplotlib as mp 
import haversine
import pandas as pd
import os
import glob

import pytz
from datetime 

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import exp


# initialise sparkContext
spark = SparkSession.builder \
    .master('local') \
    .appName('myAppName') \
    .config('spark.executor.memory', '32gb') \
    .config("spark.cores.max", "12") \
    .getOrCreate()

sc = spark.sparkContext

# using SQLContext to read parquet file
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)



#os.chdir("C:\\Users\\mbhattac\\Desktop\\coding\\data\\weather_csv")
#extension = 'csv'


#all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
##export to csv
#combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

#foo = pd.read_parquet('data\\newdata\\foo.parquet', engine='pyarrow')
# to read parquet file
drive = sqlContext.read.parquet('data\\drive\\*')
trip = sqlContext.read.parquet('data\\trip\\*')
weather = sqlContext.read.parquet('data\\weather_parquet\\*')

temp = trip.join(drive,['vehicle_id','trip_id','datetime',])
temp = temp.withColumnRenamed("long","lon")
#temp = temp.withColumnRenamed("long","combinedlon").withColumnRenamed("lat","combinedlat")

# weather = weather.withColumnRenamed("long","lon")

columns_to_drop= ['__index_level_0__','__index_level_0__']
temp = temp.drop(*columns_to_drop)
#weather=weather.withColumnRenamed("date","datetim")
weather=weather.drop('wind_ns_unit','wind_ew_unit','temperature_unit','precipitation_unit')

temp = temp.join(weather,['lat','lon']).drop('velocity')

print(temp.columns)



temp = temp.withColumn('UTCdatetime',)
#temp = temp.withColumn("PaidMonth", change_day(testdf.date))

#foo[0]['datetime'].replace(tzinfo=pytz.timezone("UTC"))
#foo  = temp.head(3)
#a[0].__getattr__('date')
#temp = temp.join(weather,['vehicle_id','trip_id','datetime',])

#temp = data.selectExpr("long as lon")
#weather.write.format("csv").save('asdasdasd')
#car = sqlContext.read.parquet('data\\newdata\\foo.parquet')

#vehicles = pd.read_csv('data\\vehicle.csv')
#vehicles=sqlContext.createDataFrame(vehicles)

#weather.write.parquet("newweather.parquet")
#foo = weather.toPandas()


#drivesample  = pd.DataFrame(drive.head(samples), columns=drive.columns)
#tripsample  = pd.DataFrame(trip.head(samples), columns=trip.columns)
#weathersample  = pd.DataFrame(weather.head(samples), columns=weather.columns)


#foo = pd.merge(drive,trip,on=['trip_id'])

#foo  = drive.select('accel_x').collect()