import json

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, functions, types
import sys
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
spark = SparkSession.builder.appName('yelp data cleaning').getOrCreate()
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

def main(input_bus, input_rev, output):
    # load business and review files, select only business_id and category fields from business file
    business = spark.read.json(input_bus).select('business_id','categories','city').createOrReplaceTempView("business")
    review = spark.read.json(input_rev).createOrReplaceTempView("review")
    # select restaurants from all businesses
    spark.sql("SELECT business_id As business_sub_id, city AS bus_city, categories FROM business WHERE categories LIKE '%Restaurant%'").createOrReplaceTempView("business_sub")
    #spark.sql("SELECT * FROM business WHERE categories NOT LIKE '%Store%' OR categories NOT LIKE '%store%' OR categories NOT LIKE '%Grocery%' OR categories NOT LIKE '%Shop%'").createOrReplaceTempView("business_sub")
    results = spark.sql("SELECT business_id, cool, date, funny, review_id, stars, text, useful, user_id, bus_city FROM review INNER JOIN business_sub ON business_id=business_sub_id")
    #results.write.csv(output)
    results_loc = results.select('business_id','user_id','bus_city').groupBy('user_id')
    results.coalesce(1).write.format('json').save(output)

if __name__ == '__main__':
    input_bus = sys.argv[1]
    input_rev = sys.argv[2]
    output = sys.argv[3]
    main(input_bus, input_rev,output)
