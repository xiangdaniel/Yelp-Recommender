import json

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, functions, types
import sys
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
spark = SparkSession.builder.appName('yelp data cleaning').getOrCreate()
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

def main(input_bus, output):
    # load business and review files, select only business_id and category fields from business file
    business = spark.read.json(input_bus).createOrReplaceTempView("business")
    # select restaurants from all businesses
    results = spark.sql("SELECT * FROM business WHERE categories LIKE '%Restaurant%'")
    #results = spark.sql("SELECT * FROM business WHERE categories NOT LIKE '%Store%' OR categories NOT LIKE '%store%' OR categories NOT LIKE '%Grocery%' OR categories NOT LIKE '%Shop%'")
    #results.write.csv(output)
    results.coalesce(1).write.format('json').save(output)









if __name__ == '__main__':
    input_bus = sys.argv[1]
    output = sys.argv[2]
    main(input_bus, output)
