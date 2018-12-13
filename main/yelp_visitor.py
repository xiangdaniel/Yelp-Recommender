from pyspark.sql import SparkSession, functions, types
import sys
from pyspark.sql.functions import split, explode
from pyspark.sql import functions
from pyspark.sql.functions import lit
from load_tools import feature_schema

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
spark = SparkSession.builder.appName('yelp recommendation for visitors cold start').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+


def main(input_bus, input_rev, input_feature, city, state, keywords, k, output):
    # 1. Recommendation for visitor
    business = spark.read.json(input_bus).repartition(50).select('business_id', 'name', 'address', 'city', 'state', 'postal_code', 'stars', 'categories')
    business = business.where((business['city'] == city) & (business['state'] == state))
    review = spark.read.json(input_rev).repartition(50).select('business_id', 'text')
    review = review.groupby('business_id').agg(functions.collect_list('text').alias('list_text'))
    review = review.withColumn('text', functions.concat_ws(' ', 'list_text')).select('business_id', functions.lower(functions.col('text')).alias('text'))
    review = review.withColumn('text', split('text', "\s+")).select('business_id', 'text')
    list_keywords = keywords.split('-')
    like_f = functions.udf(lambda col: True if set(col) & set(list_keywords) else False, types.BooleanType())
    review = review.select(review['business_id'], review['text'], like_f(review['text']).alias('check'))
    review = review.where(review['check']).select('business_id', 'text')
    condition = [business['business_id'] == review['business_id']]
    data = business.join(review, condition).select(business['business_id'], business['name'], business['address'], business['city'], business['state'], business['postal_code'], business['stars'], business['categories']).sort('stars', ascending=False)

    # 2. Join with nlp feature file
    data.createOrReplaceTempView('bus_rec_join')
    feature = spark.read.csv(input_feature, schema=feature_schema)
    feature.createOrReplaceTempView("feature")
    bus_rec_fea_join = spark.sql("""
        SELECT bus_rec_join.business_id, name, address, city, state, postal_code, categories, bus_rec_join.stars, score1, score2, score3, score4 
        FROM bus_rec_join 
        INNER JOIN feature ON bus_rec_join.business_id = feature.business_id""")

    bus_rec_fea_join = bus_rec_fea_join.withColumn("feature1", lit("food"))
    bus_rec_fea_join = bus_rec_fea_join.withColumn("feature2", lit("environment"))
    bus_rec_fea_join = bus_rec_fea_join.withColumn("feature3", lit("service"))
    bus_rec_fea_join = bus_rec_fea_join.withColumn("feature4", lit("price"))

    final_table = bus_rec_fea_join.select(
        bus_rec_fea_join['address'], bus_rec_fea_join['business_id'].alias('businessId'), bus_rec_fea_join['name'].alias('businessName'),
        bus_rec_fea_join['categories'], bus_rec_fea_join['city'],
        bus_rec_fea_join['feature1'], bus_rec_fea_join['feature2'], bus_rec_fea_join['feature3'], bus_rec_fea_join['feature4'],
        bus_rec_fea_join['postal_code'].alias('postalCode'), bus_rec_fea_join['stars'], bus_rec_fea_join['state'],
        bus_rec_fea_join['score1'], bus_rec_fea_join['score2'], bus_rec_fea_join['score3'], bus_rec_fea_join['score4'])

    final_table = final_table.sort('stars', ascending=False).limit(k)
    final_table.coalesce(1).write.option("header", "true").csv(output)


if __name__ == '__main__':
    sc = spark.sparkContext
    input_bus = sys.argv[1]
    input_rev = sys.argv[2]
    input_feature = sys.argv[3]
    city = sys.argv[4]
    state = sys.argv[5]
    keywords = sys.argv[6]
    k = int(sys.argv[7])
    output = sys.argv[8]
    main(input_bus, input_rev, input_feature, city, state, keywords, k, output)
