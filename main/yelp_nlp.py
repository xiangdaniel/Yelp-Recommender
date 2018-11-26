import sys
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
spark = SparkSession.builder.appName('yelp nlp').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+
nltk.data.path.append('/home/dxiang/nltk_data')

review_schema = types.StructType([
    types.StructField('review_id', types.StringType(), False),
    types.StructField('user_id', types.StringType(), False),
    types.StructField('business_id', types.StringType(), False),
    types.StructField('stars', types.IntegerType(), False),
    types.StructField('date', types.StringType(), False),
    types.StructField('text', types.StringType(), False),
    types.StructField('useful', types.IntegerType(), False),
    types.StructField('funny', types.IntegerType(), False),
    types.StructField('cool', types.IntegerType(), False),
])


#@functions.udf(returnType=types.ArrayType(types.StringType()))
def udf_morphy(tokens):
    if not isinstance(tokens, list):
        tokens = [tokens]
    #lemmatizer = WordNetLemmatizer()
    modified_tokens = []
    for token in tokens:
        #token = lemmatizer.lemmatize(token)
        modified_token = wn.morphy(token)
        if modified_token is None:
            continue
        modified_tokens.append(modified_token)
    return modified_tokens


#@functions.udf(returnType=types.ArrayType(types.ArrayType(types.StringType())))
def udf_pos_tag(tokens):
    if isinstance(tokens, list):
        return pos_tag(tokens)
    return pos_tag([tokens])


def classify_tokens(list_tokens):
    if not isinstance(list_tokens, list):
        list_tokens = [list_tokens]
    list_token = []
    for token in list_tokens:
        tag = wn.synsets(token)[0].pos()  # ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a'/JJ', 's', 'r', 'n', 'v'
        if tag == 'n' and pos_tag([token])[0][1] == 'NN':
            noun = wn.synsets(token)[0]
            list_hypernyms = get_parent_classes(noun)
            if token == 'food' or 'food' in list_hypernyms or 'animal' in list_hypernyms or 'fruit' in list_hypernyms or 'alcohol' in list_hypernyms or 'beverage' in list_hypernyms:
                list_token.append('food')
            elif token == 'environment' or 'location' in list_hypernyms or 'situation' in list_hypernyms or 'condition' in list_hypernyms or 'area' in list_hypernyms:
                list_token.append('environment')
            elif token == 'staff' or 'supervisor' in list_hypernyms or 'work' in list_hypernyms or 'worker' in list_hypernyms or 'consumer' in list_hypernyms:
                list_token.append('staff')
            elif token == 'price' or 'value' in list_hypernyms:
                list_token.append('price')
        else:
            if tag == 'n':
                continue
            elif tag == 'a' or tag == 's' or pos_tag([token])[0][1] == 'JJ' or pos_tag([token])[0][1] == 'JJS':
                list_token.append(token)
    return list_token


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
    i_left = -1  # left wall
    i_right = len(tokens)  # right wall
    if index_n.index(i) - 1 >= 0 and index_n[index_n.index(i) - 1] >= i_left:
        i_left = index_n[index_n.index(i) - 1]
    if index_n.index(i) + 1 < len(index_n) and index_n[index_n.index(i) + 1] <= i_right:
        i_right = index_n[index_n.index(i) + 1]
    i_best_a = -1
    min_distance = len(tokens) + 1
    for i_a in index_a:
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


def senti_score(tokens, classfication):
    index = []
    index_n = []
    index_a = []
    for i, x in enumerate(tokens):
        if x == classfication:
            index.append(i)
            index_n.append(i)
        elif wn.synsets(x)[0].pos() == 'n' and pos_tag([x])[0][1] == 'NN':
            index_n.append(i)
        else:
            index_a.append(i)
    score = 0.0
    count = 0
    if len(index) == 0 or len(index_a) == 0:
        return score
    index_a_copy = index_a[:]  # hard copy
    for i in index:
        if len(index_a_copy) == 0:
            break
        i_a = find_near_a(i, index_n, index_a_copy, tokens)
        if i_a == -1:
            continue
        #print(i_a)
        #print(tokens[i_a])
        #print(wn.synsets(tokens[i_a], pos=wn.ADJ))
        adj = wn.synsets(tokens[i_a], pos=wn.ADJ)
        score += swn.senti_synset(adj[0].name()).pos_score()
        score -= swn.senti_synset(adj[0].name()).neg_score()
        count += 1
    if count == 0:
        return score
    return score / count


def main(inputs, output):
    # 1. Load Data and Select only business_id, stars, text
    data = spark.read.json(inputs, schema=review_schema).select('business_id', 'stars', 'text')
    data = data.where(data['text'].isNotNull())  # filter reviews with no text

    # 2. ML pipeline: Tokenization (with Regular Expression) and Remove Stop Words
    regex_tokenizer = RegexTokenizer(inputCol='text', outputCol='words', pattern='[^A-Za-z]+')
    stopwords_remover = StopWordsRemover(inputCol='words',
                                         outputCol='tokens',
                                         stopWords=StopWordsRemover.loadDefaultStopWords('english'))
    # count_vectorizer = CountVectorizer(inputCol='filtered_words', outputCol='features')
    nlp_pipeline = Pipeline(stages=[regex_tokenizer, stopwords_remover])
    model = nlp_pipeline.fit(data)
    review = model.transform(data).select('business_id', 'stars', 'text', 'tokens')
    #review = review.select(review['business_id'], review['text'], udf_morphy(review['tokens']).alias('tokens'))

    # 3. Select Features
    review_pd = review.toPandas()
    review_pd['tokens'] = review_pd['tokens'].apply(udf_morphy)
    #review_pd['tokens_tag'] = review_pd['tokens'].apply(udf_pos_tag)
    review_pd['classify_tokens'] = review_pd['tokens'].apply(classify_tokens)

    # 4. Calculate Feature Weights
    review_pd['food'] = review_pd['stars'] * review_pd['classify_tokens'].apply(lambda c: senti_score(c, 'food'))
    review_pd['environment'] = review_pd['classify_tokens'].apply(lambda c: senti_score(c, 'environment'))
    review_pd['staff'] = review_pd['classify_tokens'].apply(lambda c: senti_score(c, 'staff'))
    review_pd['price'] = review_pd['classify_tokens'].apply(lambda c: senti_score(c, 'price'))

    # 5. Calculate Average Feature Weights
    review_new = spark.createDataFrame(review_pd[['business_id', 'stars', 'food', 'environment', 'staff', 'price']])
    review_new = review_new.groupby('business_id').agg(
        functions.mean('stars').alias('ave_stars'),
        functions.mean('food').alias('food'),
        functions.mean('environment').alias('environment'),
        functions.mean('staff').alias('staff'),
        functions.mean('price').alias('price')
    )

    # 6. Save
    review_new.write.csv(output, mode='overwrite')


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)