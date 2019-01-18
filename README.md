# Feature-Based Yelp Recommendation System

Feature-Based Yelp Recommendation System is the course project of CMPT 732. We developed a restaurant recommendation
system for both users and visitors using [Yelp Open Dataset](https://www.yelp.com/dataset/challenge). The Item-based Collaborative Filtering recommendation algorithm was implemented with Spark. We also researched Natural Language Processing and analyzed the reviews of users. Four features are extracted and their weights are calculated to facilitate users' understanding about the recommended restaurants. We finally created a web page to visualize results. If users log in their ids, they will get the information about the recommended restaurants based on their previous preference.


## Folder introduction
**Folder directory: main**

| File | Description |
| --- | --- |
| `yelp_recommender.py` | the Item-based Collabrative Filtering Algorithm is implemented with Spark |
| `yelp_nlp.py` | Select features and calculate the weights to facilitate recommendation |
| `yelp_visitor.py` | recommendation for visitors (cold start) |
| `load_tools.py` | Load tools |
| `yelp_business_data_clean.py` | filter out all restaurants business data |
| `yelp_review_data_clean.py` | filter out all restaurant review data |
| `yelp_recommendation_info.py` | merge business data, recommendation info and nlp feature info |

## Arguments description
yelp_recommender.py

| Arguments | Description |
| --- | --- |
| `#args1: inputs_review` | input directory for review.json |
| `#args2: inputs_business` | input directory for business.json |
| `#args3: k` | the maximum number of recommended resturants for each user |
| `#args4: output` | output directory |


yelp_nlp.py

| Arguments | Description |
| --- | --- |
| `#args1: inputs` | input directory for review.json |
| `#args2: output` | output directory |


yelp_visitor.py

| Arguments | Description |
| --- | --- |
| `#args1: input_bus` | input directory for business.json |
| `#args2: input_rev` | input directory for review.json |
| `#args3: input_feature` | input directory for nlp feature scores |
| `#args4: city` | input city |
| `#args5: state` | input state |
| `#args6: keywords` | input a list of keywords|
| `#args7: k` | input an integer |
| `#args8: output` | output directory |


yelp_business_data_clean.py

| Arguments | Description |
| --- | --- |
| `#args1: inputs_bus` | input directory for business.json |
| `#args2: output` | output directory |


yelp_review_data_clean.py

| Arguments | Description |
| --- | --- |
| `#args1: input_bus` | input directory for business.json |
| `#args2: input_rev` | input directory for review.json |
| `#args3: output` | output directory |


yelp_recommendation_info.py

| Arguments | Description |
| --- | --- |
| `#args1: input_bus` | input directory for business.json |
| `#args2: input_recommend` | input directory for our top-k recommendation |
| `#args3: input_feature` | input directory for nlp feature scores |
| `#args4: output` | output directory |


## Data Visualization

You can visit [yelp](https://github.com/cmpt732/yelp) to access details about data visualization. Web server and database (mySQL) are setup on amazon AWS. Web backend and frontend are developed with JAVA spring boot framework and AngularJS. After user input user IDs on the website,  we will return top 8 restaurants with the name and address. For these recommended restaurants, users can further filter the restaurants based on food categories :American, Asian Fusion French, Chinese, Japanese.... We also generate a histogram for each of restaurants to show their scores on the four features: Food, Service, Environment, Price, as well as the wordcloud to show the highest frequency words appear in this restaurant's reviews, so that to provide another snapshot of this restaurant. Users can based on their preference to select the restaurant without reading through tons of reviews.   
![alt text](https://github.com/xiangdaniel/Yelp-Recommender/blob/master/images/webpage.png)
