import numpy as np
import pandas as pd
from lad import TortoiseBoostRegressor
from sklearn.externals import joblib

train = pd.read_csv('train.csv', delimiter=',')
test = pd.read_csv('test.csv', delimiter=',')
test['loss'] = np.nan
joined = pd.concat([train, test])
cat_feature = [n for n in joined.columns if n.startswith('cat')]
cont_feature = [n for n in joined.columns if n.startswith('cont')]
shift = 200

for column in cat_feature:
    joined[column] = pd.factorize(joined[column].values, sort=True)[0]

train = joined[joined['loss'].notnull()]
test = joined[joined['loss'].isnull()]
ids = test['id']
Xtrain = train.drop(['loss', 'id'], 1)
Xtest = train.drop(['loss', 'id'], 1)

ytrain = train['loss']

X = Xtrain.as_matrix()
y = np.log(ytrain.values + shift)

alg = TortoiseBoostRegressor(n_estimators=25, reg_lambda=10.0,
                             max_leaf_nodes=32,
                             holdout=0.01,
                             early_stopping_rounds=5,
                             verbose=True)
alg.fit(X, y)
joblib.dump(alg, 'tortoise_boost_32_2.pkl')
