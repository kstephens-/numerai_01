import pandas as pd
import numpy as np
import os.path as osp
np.random.seed(7)

from sklearn.preprocessing import (
    MaxAbsScaler,
    normalize,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures
)
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier

from keras.models import Sequential
from keras.layers.core import Dense, Activation, MaxoutDense, Dropout
from keras.optimizers import Adagrad, SGD
from keras.layers.noise import GaussianNoise


#fs = ('nn_v0.10.csv', 'xgb_v0.03.csv', 'svm_v0.08.csv')
fs = ('nn_v0.08.csv', 'nn_v0.09.csv', 'nn_v0.11.csv', 'svm_v0.09.csv',  'xgb_v0.02.csv')

train_dir = '../data/ensemble_train'
test_dir = '../submissions'

train = pd.concat([pd.read_csv(osp.join(train_dir, f))[f.split('_')[0]] for f in fs], axis=1)
test = pd.concat(
    [pd.read_csv(osp.join(test_dir, f))
        .rename(columns={'probability': f.split('_')[0]})[f.split('_')[0]]
     for f in fs], axis=1
)

#train_labels = pd.read_csv('../data/numerai_datasets/numerai_training_data.csv')
#train['target'] = train_labels.target.values
train_full = pd.read_csv('../data/numerai_datasets/numerai_training_data.csv')
train = pd.concat([train, train_full], axis=1)

#test_index = pd.read_csv('../data/numerai_datasets/numerai_tournament_data.csv')
#test['t_id'] = test_index.t_id.values
test_full = pd.read_csv('../data/numerai_datasets/numerai_tournament_data.csv')
test = pd.concat([test, test_full], axis=1)

folds = StratifiedKFold(train.target.values,
                        n_folds=5,
                        shuffle=True,
                        random_state=4242)

# model = LogisticRegression(
#     penalty='l2',
#     C=1.0,
#     fit_intercept=True,
#     intercept_scaling=100,
#     random_state=11*13*15*17*19,
#     solver='liblinear',
#     max_iter=1000,
#     n_jobs=-1
# )
# model = KNeighborsClassifier(
#     n_neighbors=5,
#     weights='distance',
#     algorithm='auto',
#     leaf_size=30,
#     p=2,
#     metric='minkowski',
#     n_jobs=-1
# )
# model = ExtraTreesClassifier(
#     n_estimators=500,
#     criterion='gini',
#     max_depth=10,
#     min_samples_split=100,
#     min_samples_leaf=1,
#     min_weight_fraction_leaf=0,
#     n_jobs=-1,
#     random_state=58382
# )

train_train_scores = []
train_test_scores = []

submission = pd.DataFrame({'t_id': test.t_id.values})

feats = [c for c in train.columns if c.startswith('feature')]
for c in feats:

    counts = train[c].value_counts()
    train.insert(1, '{}_count'.format(c), counts[train[c]].values)
    test.insert(1, '{}_count'.format(c), counts[test[c]].values)
    test['{}_count'.format(c)].fillna(0, inplace=True)

test.fillna(0, inplace=True)


feats = train.columns.difference(['target'])
for ind, (train_index, test_index) in enumerate(folds):

    print()
    print('Fold:', ind)

    train_train = train.iloc[train_index]
    train_test = train.iloc[test_index]

    print(train_train)

    # poly = PolynomialFeatures(degree=5, interaction_only=False, include_bias=False)
    # train_train_poly = poly.fit_transform(train_train[feats])
    # train_test_poly = poly.transform(train_test[feats])

    model = Sequential()
    model.add(MaxoutDense(100, input_dim=47))
    model.add(Activation('relu'))
    model.add(GaussianNoise(0.00001))
    model.add(Dropout(0.33))

    model.add(MaxoutDense(1, input_dim=100))
    model.add(Activation('sigmoid'))

    #ada = Adagrad(lr=0.01)
    ada = SGD(lr=0.003, momentum=0.9, decay=0.001, nesterov=True)
    model.compile(optimizer=ada,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    scaler = MaxAbsScaler()
    train_train_scaled = scaler.fit_transform(train_train[feats])
    train_test_scaled = scaler.transform(train_test[feats])

    model.fit(train_train_scaled, train_train.target.values)
    train_train_pred = model.predict(train_train_scaled)
    train_test_pred = model.predict(train_test_scaled)

    train_score = log_loss(train_train.target.values, train_train_pred)
    test_score = log_loss(train_test.target.values, train_test_pred)

    #test_poly = poly.transform(test[feats])
    test_scaled = scaler.transform(test[feats])
    test_pred = model.predict(test_scaled)
    #test_pred = model.predict(test_scaled, batch_size=100)

    submission.loc[:, 'm_{}'.format(ind)] = test_pred

    print('train score:', train_score)
    train_train_scores.append(train_score)
    print('test socre:', test_score)
    train_test_scores.append(test_score)

print()
print('Avg train score:', np.mean(train_train_scores))
print('Avg test score:', np.mean(train_test_scores))
print('Test std:', np.std(train_test_scores))
print()

version = '0.01'
sub = submission[submission.columns.difference(['t_id'])].mean(axis=1)
final_submission = pd.DataFrame({'t_id': submission['t_id'].values,
                                 'probability': sub})
final_submission.to_csv('../submissions/{}_v{}.csv'.format('ensemble', version), index=False)
