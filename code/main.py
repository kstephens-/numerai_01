import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import normalize, PolynomialFeatures
from sklearn.decomposition import PCA


train = pd.read_csv('../data/numerai_datasets/numerai_training_data.csv')
test = pd.read_csv('../data/numerai_datasets/numerai_tournament_data.csv')

folds = StratifiedKFold(train.target.values,
                        n_folds=5,
                        shuffle=True,
                        random_state=7*9*11*13*15)


params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['gamma'] = 0
params['max_depth'] = 4
params['min_child_weight'] = 1
params['max_delta_step'] = 0
params['subsample'] = 1.0
#params['colsample_bytree'] = 0.9
params['colsample_bylevel'] = 0.8
params['seed'] = 4242
params['silent'] = 1

num_rounds = 350

train_train_scores = []
train_test_scores = []

submission = pd.DataFrame({'t_id': test.t_id.values})
ensemble_train = pd.DataFrame(train.index.copy())

features = train.columns[:-1]
train.insert(1, 'row_mean', train[features].mean(axis=1))
test.insert(1, 'row_mean', test[features].mean(axis=1))

# train.insert(1, 'row_max', train[features].max(axis=1))
# test.insert(1, 'row_max', test[features].max(axis=1))

# train.insert(1, 'row_min', train[features].min(axis=1))
# test.insert(1, 'row_min', test[features].min(axis=1))
feats = [c for c in train.columns if c.startswith('feature')]
for c in feats:

    train.insert(1, '{}_quant'.format(c), pd.qcut(train[c], 500, labels=False))
    test.insert(1, '{}_quant'.format(c), pd.qcut(test[c], 500, labels=False))

    counts = train['{}_quant'.format(c)].value_counts()
    train.insert(1, '{}_count'.format(c), counts[train['{}_quant'.format(c)]].values / counts.sum())
    test.insert(1, '{}_count'.format(c), counts[test['{}_quant'.format(c)]].values / counts.sum())
    test['{}_count'.format(c)].fillna(0, inplace=True)

    #train.insert(1, '{}_quant'.format(c), pd.qcut(train[c], 50, labels=False))
    #test.insert(1, '{}_quant'.format(c), pd.qcut(test[c], 50, labels=False))

train.drop(feats, axis=1, inplace=True)
test.drop(feats, axis=1, inplace=True)

#train.insert(1, 'feature_sum', (train[train.columns[:-1]] == 0).sum(axis=1))
#test.insert(1, 'feature_sum', (test[test.columns[1:-1]] == 0).sum(axis=1))
# train.insert(1, 'feature10_feature11', train['feature10'] + train['feature11'] + train['feature21'])
# test.insert(1, 'feature10_feature11', test['feature10'] + test['feature11'] + test['feature21'])

# train.drop(['feature10', 'feature11', 'feature21'], axis=1, inplace=True)
# test.drop(['feature10', 'feature11', 'feature21'], axis=1, inplace=True)
# feats = train.columns[:-1]
# for i in range(len(feats)-1):
#     train.insert(1, '{}_{}'.format(feats[i+1], feats[i]), train[feats[i+1]] - train[feats[i]])
#     test.insert(1, '{}_{}'.format(feats[i+1], feats[i]), test[feats[i+1]] - test[feats[i]])

features = train.columns[:-1]
test_features = test.columns[1:]
for ind, (train_index, test_index) in enumerate(folds):

    print()
    print('Fold:', ind)

    train_train = train.iloc[train_index]
    train_test = train.iloc[test_index]

    print('train shape:', train_train.shape)
    print('test shape:', train_test.shape)

    poly = PolynomialFeatures(include_bias=False)
    train_train_poly = poly.fit_transform(train_train[features])
    train_test_poly = poly.transform(train_test[features])

    pca = PCA(n_components=2)
    x_train_train_poly = pca.fit_transform(train_train_poly)
    x_train_test_poly = pca.transform(train_test_poly)

    train_train.insert(1, 'pca_poly1', x_train_train_poly[:, 0])
    train_train.insert(1, 'pca_poly2', x_train_train_poly[:, 1])

    train_test.insert(1, 'pca_poly1', x_train_test_poly[:, 0])
    train_test.insert(1, 'pca_poly2', x_train_test_poly[:, 1])

    features = train.columns[:-1]
    dtrain_train = xgb.DMatrix(train_train[features],
                               train_train.target.values,
                               silent=True)
    dtrain_test = xgb.DMatrix(train_test[features],
                              train_test.target.values,
                              silent=True)

    watchlist = [(dtrain_train, 'train'), (dtrain_test, 'test')]
    model = xgb.train(params, dtrain_train, num_rounds,
                      evals=watchlist, early_stopping_rounds=50,
                      verbose_eval=False)

    train_train_pred = model.predict(dtrain_train, ntree_limit=model.best_iteration)
    train_test_pred = model.predict(dtrain_test, ntree_limit=model.best_iteration)

    # lr = LogisticRegression(penalty='l2', dual=False, C=0.01, intercept_scaling=30.0, solver='lbfgs', random_state=1234)
    # lr.fit(train_train_pred.reshape(-1, 1), train_train.target.values)
    # train_train_pred_cal = lr.predict_proba(train_train_pred.reshape(-1, 1))
    # train_test_pred_cal = lr.predict_proba(train_test_pred.reshape(-1, 1))

    # ir = IsotonicRegression(out_of_bounds='clip')
    # ir.fit(train_train_pred, train_train.target.values)
    # train_train_pred_cal = ir.transform(train_train_pred)
    # train_test_pred_cal = ir.transform(train_test_pred)

    train_score = log_loss(train_train.target.values, train_train_pred)
    test_score = log_loss(train_test.target.values, train_test_pred)

    test_poly = poly.transform(test[test_features])
    x_test = pca.transform(test_poly)

    try:
        test.insert(1, 'pca_poly1', x_test[:, 0])
        test.insert(1, 'pca_poly2', x_test[:, 1])
    except ValueError:
        test['pca_poly1'] = x_test[:, 0]
        test['pca_poly2'] = x_test[:, 1]

    dtest = xgb.DMatrix(test[features], silent=True)
    test_pred = model.predict(dtest, ntree_limit=model.best_iteration)

    ensemble_train.loc[train_test.index, 'xgb'] = train_test_pred
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

version = '0.03'
sub = submission[submission.columns.difference(['t_id'])].mean(axis=1)
final_submission = pd.DataFrame({'t_id': submission['t_id'].values,
                                 'probability': sub})
final_submission.to_csv('../submissions/{}_v{}.csv'.format('xgb', version), index=False)

ensemble_train.to_csv('../data/ensemble_train/{}_v{}.csv'.format('xgb', version), index=False)
