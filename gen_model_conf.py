#!/home/rsebastian/.anaconda3/envs/ansible/bin/python

import json

conf = []

# models = ['xgbClassifier', 'xgbOrdinalClassifier', 'xgbCoarseGrainedClassifier']
models = ['xgbClassifier']
# vec_types = [
#     'kw', 'babelkw', 'wikikw', 'st', 'tfidf',
#     'babelkw+wikikw', 'st+wikikw', 'st+babelkw',
#     'babelkw+kw', 'wikikw+kw', 'wikikw+babelkw+kw']
vec_types = ['kw', 'babelkw', 'wikikw']
lander = ['baden-wuerttemberg', 'bayern', 'berlin', 'brandenburg', 'bremen',
       'hamburg', 'hessen', 'mecklenburg-vorpommern',
       'nordrhein-westfalen', 'saarland', 'sachsen', 'sachsen-anhalt',
       'schleswig-holstein', 'thueringen', 'niedersachsen',
       'rheinland-pfalz']
# filtered = [True, False]
filtered_values = [False]

for model in models:
    for vec_type in vec_types:
        for land in lander:
            for filtered in filtered_values:
                conf.append({
                    'name': model,
                    'args': ['{{nthreads}}'],
                    'vec_type': vec_type,
                    'filtered': filtered,
                    'land': land
                })

print(json.dumps(conf))
