import json
import pandas as pd
import numpy as np
from itertools import chain
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import split, explode
from pyspark.sql import functions as F
from pyspark.sql.functions import lit
import sys
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
spark = SparkSession.builder.appName('yelp recommendation info').getOrCreate()
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

feature_schema = types.StructType([
    types.StructField('business_id', types.StringType(), False),
    types.StructField('stars', types.FloatType(), False),
    types.StructField('score1', types.FloatType(), False), #food
    types.StructField('score2', types.FloatType(), False), #environment
    types.StructField('score3', types.FloatType(), False), #service
    types.StructField('score4', types.FloatType(), False), #price
])

recommend_schema = types.StructType([
    types.StructField('user_id', types.StringType(), False),
    types.StructField('business_id', types.StringType(), False),
    types.StructField('score', types.FloatType(), False),
])

def add_one(num):
    num = num + 1
    return num

def main(input_bus, input_recommend,input_feature,output):
    # load business and review files, select only business_id and category fields from business file
    business = spark.read.json(input_bus).select('business_id','name','address','city','state','stars','postal_code','categories').createOrReplaceTempView("business")
    recommend = spark.read.csv(input_recommend, schema=recommend_schema).createOrReplaceTempView("recommend")
    feature = spark.read.csv(input_feature, schema=feature_schema).createOrReplaceTempView("feature")
    spark.sql("SELECT recommend.business_id,name,address,city,state,stars,postal_code,categories,user_id,score FROM business INNER JOIN recommend ON business.business_id == recommend.business_id").createOrReplaceTempView("bus_rec_join")
    bus_rec_fea_join = spark.sql("SELECT bus_rec_join.business_id,name,address,city,state,postal_code,categories,user_id,feature.stars,score1,score2,score3,score4 FROM bus_rec_join INNER JOIN feature ON bus_rec_join.business_id == feature.business_id")
    bus_rec_fea_join = bus_rec_fea_join.withColumn("feature1", lit("food"))
    bus_rec_fea_join = bus_rec_fea_join.withColumn("feature2", lit("environment"))
    bus_rec_fea_join = bus_rec_fea_join.withColumn("feature3", lit("service"))
    bus_rec_fea_join = bus_rec_fea_join.withColumn("feature4", lit("price"))
    bus_rec_fea_join = bus_rec_fea_join.orderBy(['user_id','stars'],ascending=[1, 0])
    bus_rec_fea_join = bus_rec_fea_join.withColumn("id", F.monotonically_increasing_id())
    #bus_rec_fea_join['id'] = bus_rec_fea_join['id'].apply(index)
    ###
    final_table = bus_rec_fea_join.select(add_one(bus_rec_fea_join['id']).alias('id'),bus_rec_fea_join['address'],bus_rec_fea_join['business_id'].alias('businessId'),bus_rec_fea_join['name'].alias('businessName'),\
    bus_rec_fea_join['categories'],bus_rec_fea_join['city'],bus_rec_fea_join['feature1'],bus_rec_fea_join['feature2'],bus_rec_fea_join['feature3'],bus_rec_fea_join['feature4'],bus_rec_fea_join['postal_code'].alias('postalCode'),\
    bus_rec_fea_join['stars'],bus_rec_fea_join['state'], bus_rec_fea_join['user_id'].alias('userId'), \
    bus_rec_fea_join['score1'],bus_rec_fea_join['score2'],bus_rec_fea_join['score3'],bus_rec_fea_join['score4'])
    ###

    #final_table.coalesce(1).write.format('json').save(output)
    final_table.coalesce(1).write.option("header", "true").csv(output)

if __name__ == '__main__':
    input_bus = sys.argv[1]
    input_recommend = sys.argv[2]
    input_feature = sys.argv[3]
    output = sys.argv[4]
    main(input_bus,input_recommend ,input_feature,output)
