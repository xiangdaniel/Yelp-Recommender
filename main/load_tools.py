from pyspark.sql import SparkSession, functions, types

recommender_schema = types.StructType([
    types.StructField('user_id', types.StringType(), False),
    types.StructField('recommendation_business_id', types.StringType(), False),
    types.StructField('predicted_stars', types.FloatType(), False)
])

feature_schema = types.StructType([
    types.StructField('business_id', types.StringType(), False),
    types.StructField('stars', types.FloatType(), False),
    types.StructField('score1', types.FloatType(), False),  # food
    types.StructField('score2', types.FloatType(), False),  # environment
    types.StructField('score3', types.FloatType(), False),  # service
    types.StructField('score4', types.FloatType(), False),  # price
])

business_schema = types.StructType([
    types.StructField('business_id', types.StringType(), False),
    types.StructField('name', types.StringType(), False),
    types.StructField('neighborhood', types.StringType(), False),
    types.StructField('address', types.IntegerType(), False),
    types.StructField('city', types.StringType(), False),
    types.StructField('state', types.StringType(), False),
    types.StructField('postal code', types.StringType(), False),
    types.StructField('latitude', types.FloatType(), False),
    types.StructField('longitude', types.FloatType(), False),
    types.StructField('stars', types.FloatType(), False),
    types.StructField('review_count', types.IntegerType(), False)
])

review_schema = types.StructType([
    types.StructField('review_id', types.StringType(), False),
    types.StructField('user_id', types.StringType(), False),
    types.StructField('business_id', types.StringType(), False),
    types.StructField('stars', types.IntegerType(), False),
    types.StructField('date', types.StringType(), False),
    types.StructField('text', types.StringType(), False),
    types.StructField('useful', types.IntegerType(), False),
    types.StructField('funny', types.IntegerType(), False),
    types.StructField('cool', types.IntegerType(), False)
])
