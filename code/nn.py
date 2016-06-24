import pandas as pd
import numpy as np
np.random.seed(3*5*7*9*11*13)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, MaxoutDense
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1l2
from keras.optimizers import Adagrad, Adadelta, SGD
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import LeakyReLU, PReLU

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    Normalizer,
    PolynomialFeatures
)
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss


train = pd.read_csv('../data/numerai_datasets/numerai_training_data.csv')
test = pd.read_csv('../data/numerai_datasets/numerai_tournament_data.csv')

folds = StratifiedKFold(train.target.values,
                        n_folds=5,
                        shuffle=True,
                        random_state=42)

# model = Sequential()
# model.add(MaxoutDense(100, input_dim=42))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))

# model.add(MaxoutDense(1, input_dim=100))
# model.add(Activation('sigmoid'))

# #ada = Adagrad(lr=0.001)
# ada = SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=True)
# model.compile(optimizer=ada,
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

train_train_scores = []
train_test_scores = []

submission = pd.DataFrame({'t_id': test.t_id.values})
ensemble_train = pd.DataFrame(train.index.copy())

# features = train.columns[:-1]
# train.insert(1, 'row_max', train[features].max(axis=1))
# test.insert(1, 'row_max', test[features].max(axis=1))

# train.insert(1, 'row_min', train[features].min(axis=1))
# test.insert(1, 'row_min', test[features].min(axis=1))

feats = [c for c in train.columns if c.startswith('feature')]
for c in feats:

    counts = train[c].value_counts()
    train.insert(1, '{}_count'.format(c), counts[train[c]].values)
    test.insert(1, '{}_count'.format(c), counts[test[c]].values)
    test['{}_count'.format(c)].fillna(0, inplace=True)

    #train.insert(1, '{}_diff'.format(c), train[c] / train[c].max(axis=0))
    #test.insert(1, '{}_diff'.format(c), test[c] / test[c].max(axis=0))

# train.insert(1, 'row_max', train[feats].max(axis=1))
# test.insert(1, 'row_max', test[feats].max(axis=1))

# train.insert(1, 'row_min', train[feats].min(axis=1))
# test.insert(1, 'row_min', test[feats].min(axis=1))

#train.insert(1, 'row_std', train[feats].std(axis=1))
#test.insert(1, 'row_std', test[feats].std(axis=1))

#train.insert(1, 'row_mean', train[feats].mean(axis=1))
#test.insert(1, 'row_mean', test[feats].mean(axis=1))

#train.insert(1, 'row_median', train[feats].median(axis=1))
#test.insert(1, 'row_median', test[feats].median(axis=1))

#train.insert(1, 'row_mad', train[feats].mad(axis=1))
#test.insert(1, 'row_mad', test[feats].mad(axis=1))
#train.insert(1, 'max_diff', train[feats].max(axis=1) - train[feats].min(axis=1))
#test.insert(1, 'max_diff', test[feats].max(axis=1) - test[feats].min(axis=1))
# for i in range(len(feats)-1):
#     train.insert(1, '{}_rel_change'.format(feats[i]), train[feats[i]] * train[feats[i+1]])
#     test.insert(1, '{}_rel_change'.format(feats[i]), test[feats[i]] * test[feats[i+1]])
# for col in train[feats].columns:
#     train.insert(1, '{}_divergence'.format(col), (train[col] - train[col].mean()))
#     test.insert(1, '{}_divergence'.format(col), (test[col] - test[col].mean()))

features = train.columns[:-1]
for ind, (train_index, test_index) in enumerate(folds):

    print()
    print('Fold:', ind)

    train_train = train.iloc[train_index]
    train_test = train.iloc[test_index]

    # poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    # train_train_poly = poly.fit_transform(train_train[features])
    # train_test_poly = poly.transform(train_test[features])
    model = Sequential()
    model.add(MaxoutDense(100, input_dim=42))
    model.add(Activation('relu'))
    model.add(GaussianNoise(0.00001))
    model.add(Dropout(0.3))

    model.add(MaxoutDense(1, input_dim=100))
    model.add(Activation('sigmoid'))

    #ada = Adagrad(lr=0.001)
    ada = SGD(lr=0.0003, momentum=0.9, decay=0.0001, nesterov=True)
    model.compile(optimizer=ada,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    scaler = MaxAbsScaler()
    train_train_scaled = scaler.fit_transform(train_train[features])
    train_test_scaled = scaler.transform(train_test[features])

    model.fit(train_train_scaled, train_train.target.values, nb_epoch=150, batch_size=100)

    train_train_pred = model.predict(train_train_scaled, batch_size=100)
    train_test_pred = model.predict(train_test_scaled, batch_size=100)

    train_score = log_loss(train_train.target.values, train_train_pred)
    test_score = log_loss(train_test.target.values, train_test_pred)

    #test_poly = poly.transform(test[features])
    test_scaled = scaler.transform(test[features])
    test_pred = model.predict(test_scaled, batch_size=100)

    ensemble_train.loc[train_test.index, 'nn'] = train_test_pred
    submission.loc[:, 'm_{}'.format(ind)] = test_pred

    print('train score:', train_score)
    train_train_scores.append(train_score)
    print('test score:', test_score)
    train_test_scores.append(test_score)


print()
print('Avg train score:', np.mean(train_train_scores))
print('Avg test score:', np.mean(train_test_scores))
print('Test std:', np.std(train_test_scores))
print()

version = '0.12'
sub = submission[submission.columns.difference(['t_id'])].mean(axis=1)
final_submission = pd.DataFrame({'t_id': submission['t_id'].values,
                                 'probability': sub})
final_submission.to_csv('../submissions/{}_v{}.csv'.format('nn', version), index=False)

ensemble_train.to_csv('../data/ensemble_train/{}_v{}.csv'.format('nn', version), index=False)
