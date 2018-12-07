"""
Yelp Recommender
Implement the Item-based Collaborative Filtering
Copyright (c) 2018 Daniel D Xiang
Licensed under the MIT License
Written by Daniel Xiang
"""

import functools
import json
import operator
import heapq

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from load_tools import recommender_schema

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
    if isinstance(kv[1], list):
        list_business_stars = kv[1]
    else:
        list_business_stars = [kv[1]]
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
    if isinstance(kv[1][1], list):
        list_user_rating = kv[1][1]
    else:
        list_user_rating = [kv[1][1]]
    for user_rating in list_user_rating:
        yield (user_rating[0], businessA), relation * user_rating[1]


def get_loc(line):
    businessId = line['business_id']
    city = line['city']
    state = line['state']
    return businessId, city + state


def get_user(line):
    userId = line['user_id']
    businessId = line['business_id']
    return businessId, userId


def user_city_map(kv):
    # business, (list[reviewed_user], business_city)
    city = kv[1][1]
    if isinstance(kv[1][0], list):
        list_user = kv[1][0]
    else:
        list_user = [kv[1][0]]
    for user in list_user:
        yield user, city


def frequent_city(visited_cities):
    dict = {}
    for city in visited_cities:
        if city not in dict:
            dict[city] = 1
        else:
            dict[city] += 1
    p = []
    for key, value in dict.items():
        p.append((value, key))
    sorted(p, key=functools.cmp_to_key(cmp_city))
    return p[0][1]


def cmp_city(a, b):
    if a[0] > b[0] or a[0] == b[0] and a[1] < b[1]:
        return -1
    elif a[0] == b[0] and a[1] == b[1]:
        return 0
    else:
        return 1


def find_user_city(kv):
    user = kv[0]
    if isinstance(kv[1], list):
        visited_cities = kv[1]
        return user, frequent_city(visited_cities)
    else:
        return user, kv[1]


def flag_map(line):
    userId = line['user_id']
    businessId = line['business_id']
    return userId, businessId


def recommender(kv, k):
    # USER, (list[(BUSINESS, PREDICTED_STARTS, BUSINESS_CITY), (list[business], user_city))
    heap = []
    userId = kv[0]
    if isinstance(kv[1][0], list):
        list_business_rating_businessCity = kv[1][0]
    else:
        list_business_rating_businessCity = [kv[1][0]]
    userCity = kv[1][1][1]
    if isinstance(kv[1][1][0], list):
        list_ratedBusiness = kv[1][1][0]
    else:
        list_ratedBusiness = [kv[1][1][0]]
    for business_rating_businessCity in list_business_rating_businessCity:
        businessId = business_rating_businessCity[0]
        rating = business_rating_businessCity[1]
        businessCity = business_rating_businessCity[2]
        if businessId in list_ratedBusiness or businessCity != userCity:
            continue
        heapq.heappush(heap, (rating, businessId))
        if len(heap) > k:
            heapq.heappop(heap)
    while len(heap) > 0:
        rating, businessId = heapq.heappop(heap)
        yield userId, businessId, rating


def main(inputs_review, inputs_business, k, output):
    # 1. LOAD DATA
    review = sc.textFile(inputs_review).repartition(50).map(json.loads).cache()
    business = sc.textFile(inputs_business).repartition(50).map(json.loads)

    # 2. BUILD Co-Occurrence Matrix: BUSINESSB, (BUSINESSA, RELATION)
    dataByUser = review.map(rating_map).reduceByKey(add_pairs)
    co_occurrence = dataByUser.flatMap(co_occurrence_generator).reduceByKey(operator.add)
    norm_co_occurrence = co_occurrence.map(norm_map).reduceByKey(norm_add).flatMap(norm_cooccurrence)

    # 3. BUILD RATING MATRIX: BUSINESS, list[(USER, STARS)]
    rating = review.map(rating_generator).reduceByKey(add_pairs)

    # 4. Matrix Multiplication: USER, list[(BUSINESS, PREDICTED_STARTS, BUSINESS_CITY)]
    business_loc = business.map(get_loc).cache()  # business, city
    userBusiness_rating = norm_co_occurrence.join(rating).flatMap(multiplication).reduceByKey(operator.add)  # (user, business), predicted_starts
    business_userRating = userBusiness_rating.map(lambda kv: (kv[0][1], (kv[0][0], kv[1])))  # business, (user, predicted_starts)
    business_userRating_city = business_userRating.join(business_loc)  # business, ((user, predicted_starts), business_city)
    user_businessRatingCity = business_userRating_city.map(lambda kv: (kv[1][0][0], (kv[0], kv[1][0][1], kv[1][1]))).reduceByKey(add_pairs)

    # 5. RECOMMENDER MODEL
    business_user = review.map(get_user).reduceByKey(add_pairs)  # business, list[reviewed_user]
    user_city = business_user.join(business_loc).flatMap(user_city_map).reduceByKey(add_pairs).map(find_user_city)  # user, user_city
    flag = review.map(flag_map).reduceByKey(add_pairs).join(user_city)  # user, (list[business], user_city)
    user_topk = user_businessRatingCity.join(flag).flatMap(lambda c: recommender(c, k))
    sqlContext.createDataFrame(user_topk, schema=recommender_schema).sort('user_id').write.csv(output, mode='overwrite')


if __name__ == '__main__':
    conf = SparkConf().setAppName('yelp recommender')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    assert sc.version >= '2.3'  # make sure we have Spark 2.3+

    inputs_review = sys.argv[1]
    inputs_business = sys.argv[2]
    k = int(sys.argv[3])
    output = sys.argv[4]

    main(inputs_review, inputs_business, k, output)
