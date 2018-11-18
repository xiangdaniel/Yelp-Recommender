import json
import operator
import heapq

from pyspark import SparkConf, SparkContext
import sys
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+


def rating_map(line):
    userId = line['user_id']
    businessId = line['business_id']
    stars = line['stars']
    return userId, (businessId, stars)


def add_pairs(a, b):
    l1 = list_flatten([a])
    l2 = list_flatten([b])
    return l1 + l2


def list_flatten(l):
    out = []
    for item in l:
        if isinstance(item, list):
            out.extend(list_flatten(item))
        else:
            out.append(item)
    return out


def co_occurrence_generator(kv):
    list_business_stars = list_flatten([kv[1]])
    for item1 in list_business_stars:
        business1 = item1[0]
        for item2 in list_business_stars:
            business2 = item2[0]
            yield (business1, business2), 1


def norm_map(kv):
    return kv[0][0], ((kv[0][1], kv[1]), kv[1])


def norm_add(a, b):
    l1 = list_flatten([a[0]])
    l2 = list_flatten([b[0]])
    return l1 + l2, a[1] + b[1]


def norm_cooccurrence(kv):
    if isinstance(kv[1][0], list):
        for item in kv[1][0]:
            print(type(item[1]))
            print(type(kv[1][1]))
            yield item[0], (kv[0], item[1] / kv[1][1])
    else:
        yield kv[1][0][0], (kv[0], 1)


def rating_generator(line):
    userId = line['user_id']
    businessId = line['business_id']
    stars = line['stars']
    return businessId, (userId, stars)


def multiplication(kv):
    businessB = kv[0]
    businessA = kv[1][0][0]
    relation = kv[1][0][1]
    list_user_rating = list_flatten([kv[1][1]])
    for user_rating in list_user_rating:
        yield (user_rating[0], businessA), relation * user_rating[1]


def flag_map(line):
    userId = line['user_id']
    businessId = line['business_id']
    return userId, businessId


def recommender(user_businessRating, flag_bcast, k):
    heap = []
    userId = user_businessRating[0]
    list_business_rating = list_flatten([user_businessRating[1]])
    list_ratedBusiness = list_flatten([flag_bcast.value.get(userId)])
    for business_rating in list_business_rating:
        businessId = business_rating[0]
        rating = business_rating[1]
        if businessId in list_ratedBusiness:
            continue
        heapq.heappush(heap, (rating, businessId))
        if len(heap) > k:
            heapq.heappop(heap)
    list = []
    while len(heap) > 0:
        rating, businessId = heapq.heappop(heap)
        list.insert(0, (businessId, rating))
    return userId, list


def average(kv):
    return kv[0], kv[1][1] / kv[1][0]


def get_key(kv):
    return kv[0]


def main(inputs, k, output):
    # 1. LOAD DATA
    review = sc.textFile(inputs).repartition(100).map(json.loads).cache()

    # 2. BUILD Co-Occurrence Matrix: BUSINESSB, (BUSINESSA, RELATION)
    dataByUser = review.map(rating_map).reduceByKey(add_pairs)
    co_occurrence = dataByUser.flatMap(co_occurrence_generator).reduceByKey(operator.add)
    norm_co_occurrence = co_occurrence.map(norm_map).reduceByKey(norm_add).flatMap(norm_cooccurrence)

    # 3. BUILD RATING MATRIX: BUSINESS, list[(USER, STARS)]
    rating = review.map(rating_generator).reduceByKey(add_pairs)

    # 4. Matrix Multiplication
    userBusiness_rating = norm_co_occurrence.join(rating).flatMap(multiplication).reduceByKey(operator.add)
    user_businessRating = userBusiness_rating.map(lambda kv: (kv[0][0], (kv[0][1], kv[1]))).reduceByKey(add_pairs)

    # 5. RECOMMENDER MODEL
    flag = review.map(flag_map).reduceByKey(add_pairs)
    flag_bcast = sc.broadcast(dict(flag.collect()))
    user_topk = user_businessRating.map(lambda c: recommender(c, flag_bcast, k)).sortBy(get_key).map(json.dumps)
    user_topk.saveAsTextFile(output)


if __name__ == '__main__':
    conf = SparkConf().setAppName('yelp recommender')
    sc = SparkContext(conf=conf)
    assert sc.version >= '2.3'  # make sure we have Spark 2.3+

    inputs = sys.argv[1]
    k = int(sys.argv[2])
    output = sys.argv[3]

    main(inputs, k, output)
