import numpy as np
import pandas as pd
from tortoiseboost import TortoiseBoostRegressor
from sklearn.externals import joblib

train = pd.read_csv('input/train.csv', delimiter=',')
cat_feature = [n for n in train.columns if n.startswith('cat')]
cont_feature = [n for n in train.columns if n.startswith('cont')]
shift = 200

for column in cat_feature:
    train[column] = pd.factorize(train[column].values, sort=True)[0]
Xtrain = train.drop(['loss', 'id'], 1)
ytrain = train['loss']

X = Xtrain.as_matrix()
y = np.log(ytrain.values + shift)

alg = TortoiseBoostRegressor(n_estimators=10, reg_alpha=10.0,
                             max_leaf_nodes=8)
alg.fit(X, y)
joblib.dump(alg, 'tortoise_boost_32_2.pkl')
