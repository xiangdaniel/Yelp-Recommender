"""
Yelp Recommender
Select features and calculate the weights to facilitate recommendation
Copyright (c) 2018 Daniel D Xiang
Licensed under the MIT License
Written by Daniel Xiang
"""

import sys
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from load_tools import business_schema, review_schema

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
spark = SparkSession.builder.appName('yelp nlp').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+
nltk.data.path.append('/home/dxiang/nltk_data')


#@functions.udf(returnType=types.ArrayType(types.StringType()))
def py_morphy(tokens):
    from nltk.corpus import wordnet as wn
    nltk.data.path.append('/home/dxiang/nltk_data')
    if not isinstance(tokens, list):
        tokens = [tokens]
    modified_tokens = []
    for token in tokens:
        modified_token = wn.morphy(token)
        if modified_token is None:
            continue
        modified_tokens.append(modified_token)
    return modified_tokens


udf_morphy = functions.udf(py_morphy, returnType=types.ArrayType(types.StringType()))


def classify_tokens(list_tokens):
    from nltk.corpus import wordnet as wn
    nltk.data.path.append('/home/dxiang/nltk_data')
    if not isinstance(list_tokens, list):
        list_tokens = [list_tokens]
    list_token = []
    for token in list_tokens:
        tag = wn.synsets(token)[0].pos()  # ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a/JJ', 's', 'r', 'n', 'v'
        if tag == 'n' and pos_tag([token])[0][1] == 'NN':
            noun = wn.synsets(token)[0]
            list_hypernyms = get_parent_classes(noun)
            if token == 'food' or token == 'drink' or 'food' in list_hypernyms or 'animal' in list_hypernyms or 'fruit' in list_hypernyms or 'alcohol' in list_hypernyms or 'beverage' in list_hypernyms:
                list_token.append('food')
            elif token == 'environment' or 'location' in list_hypernyms or 'situation' in list_hypernyms or 'condition' in list_hypernyms or 'area' in list_hypernyms:
                list_token.append('environment')
            elif token == 'staff' or 'supervisor' in list_hypernyms or 'work' in list_hypernyms or 'worker' in list_hypernyms or 'consumer' in list_hypernyms:
                list_token.append('service')
            elif token == 'price' or 'value' in list_hypernyms:
                list_token.append('price')
        else:
            if tag == 'a' or tag == 's' or pos_tag([token])[0][1] == 'JJ' or pos_tag([token])[0][1] == 'JJS':
                list_token.append(token)
    return list_token


udf_classify_tokens = functions.udf(classify_tokens, returnType=types.ArrayType(types.StringType()))


def get_parent_classes(synset):
    list_hypernyms = []
    while True:
        try:
            synset = synset.hypernyms()[-1]
            list_hypernyms.append(synset.lemmas()[0].name())
        except IndexError:
            break
    return list_hypernyms


def find_near_a(i, index_n, index_a, tokens):
    from nltk.corpus import wordnet as wn
    i_left = -1  # left wall
    i_right = len(tokens)  # right wall
    if index_n.index(i) - 1 >= 0 and index_n[index_n.index(i) - 1] >= i_left:
        i_left = index_n[index_n.index(i) - 1]
    if index_n.index(i) + 1 < len(index_n) and index_n[index_n.index(i) + 1] <= i_right:
        i_right = index_n[index_n.index(i) + 1]
    i_best_a = -1
    min_distance = len(tokens) + 1
    for i_a in index_a[::-1]:
        adj = wn.synsets(tokens[i_a], pos=wn.ADJ)
        if len(adj) == 0:
            index_a.remove(i_a)
            continue
        if i_left < i_a < i_right and abs(i_a - i) < min_distance:
            i_best_a = i_a
            min_distance = abs(i_a - i)
    if i_best_a != -1:
        index_a.remove(i_best_a)
    return i_best_a


def senti_score(tokens):
    from nltk.corpus import wordnet as wn
    from nltk.corpus import sentiwordnet as swn
    nltk.data.path.append('/home/dxiang/nltk_data')
    classfications = ['food', 'environment', 'service', 'price']
    index_n = []
    index_a = []
    for i, x in enumerate(tokens):
        if wn.synsets(x)[0].pos() == 'n' and pos_tag([x])[0][1] == 'NN':
            index_n.append(i)
        else:
            index_a.append(i)
    scores = [0.0, 0.0, 0.0, 0.0]  # (score_food, score_environment, score_staff, score_price)
    counts = [0, 0, 0, 0]  # (count_food, count_environment, count_staff, count_price)
    if len(index_n) == 0 or len(index_a) == 0:
        return scores
    for i in index_n:
        if len(index_a) == 0:
            break
        i_a = find_near_a(i, index_n, index_a, tokens)
        if i_a == -1:
            continue
        i_class = classfications.index(tokens[i])
        adj = wn.synsets(tokens[i_a], pos=wn.ADJ)
        scores[i_class] += swn.senti_synset(adj[0].name()).pos_score()
        #scores[i_class] -= swn.senti_synset(adj[0].name()).neg_score()
        counts[i_class] += 1
    for i in range(4):
        if counts[i] != 0:
            scores[i] /= counts[i]
    return scores


udf_senti_score = functions.udf(senti_score, returnType=types.ArrayType(types.FloatType()))


def main(inputs, output):
    # 1. Load Data and Select only business_id, stars, text
    data = spark.read.json(inputs, schema=review_schema).repartition(50).select('business_id', 'stars', 'text')
    data = data.where(data['text'].isNotNull())  # filter reviews with no text

    # 2. ML pipeline: Tokenization (with Regular Expression) and Remove Stop Words
    regex_tokenizer = RegexTokenizer(inputCol='text', outputCol='words', pattern='[^A-Za-z]+')
    stopwords_remover = StopWordsRemover(inputCol='words',
                                         outputCol='tokens',
                                         stopWords=StopWordsRemover.loadDefaultStopWords('english'))
    # count_vectorizer = CountVectorizer(inputCol='filtered_words', outputCol='features')
    nlp_pipeline = Pipeline(stages=[regex_tokenizer, stopwords_remover])
    model = nlp_pipeline.fit(data)
    review = model.transform(data).select('business_id', 'stars', 'tokens')

    # 3. Select Features
    review = review.select(review['business_id'], review['stars'], udf_morphy(review['tokens']).alias('tokens'))
    review = review.where(functions.size(review['tokens']) > 0)
    review = review.withColumn('classify_tokens', udf_classify_tokens(review['tokens']))

    # 4. Calculate Feature Weights
    review = review.withColumn('feature_weights', udf_senti_score(review['classify_tokens']))
    review = review.withColumn('food', review['stars'] * review['feature_weights'][0])
    review = review.withColumn('environment', review['stars'] * review['feature_weights'][1])
    review = review.withColumn('service', review['stars'] * review['feature_weights'][2])
    review = review.withColumn('price', review['stars'] * review['feature_weights'][3])

    # 5. Calculate Average Feature Weights
    review_new = review.select('business_id', 'stars', 'food', 'environment', 'service', 'price')
    review_new = review_new.groupby('business_id').agg(
        functions.mean('stars').alias('ave_stars'),
        functions.mean('food').alias('food'),
        functions.mean('environment').alias('environment'),
        functions.mean('service').alias('service'),
        functions.mean('price').alias('price')
    )

    # 6. Save
    review_new.write.csv(output, mode='overwrite')


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)
