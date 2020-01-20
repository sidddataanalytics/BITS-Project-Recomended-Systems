#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created at 8/20/2019
__author__ = 212577071
Usage: 
"""

import pandas as pd
from surprise import Reader, SVD
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import cross_validate
from surprise import accuracy
from model.svd_recommendation.prepare_data import genearte_score_df
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering
import matplotlib.pyplot as plt 
%matplotlib inline

MAXIMUM_REPORT_CANDIDATES = 5


class MyDataSet(DatasetAutoFolds):
    def __init__(self, df, reader=Reader(line_format='user item rating', rating_scale=(1, 5))):
        super().__init__(reader=reader, df=df)
        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in zip(df['user_sso'], df['report_id'], df['score'])]
        self.reader = reader


# noinspection DuplicatedCode
def get_top_n(predictions, n=MAXIMUM_REPORT_CANDIDATES) -> pd.DataFrame:
    uid = [uid for (uid, iid, true_r, est, _) in predictions]
    iid = [iid for (uid, iid, true_r, est, _) in predictions]
    est = [est for (uid, iid, true_r, est, _) in predictions]

    recommendation_df = pd.DataFrame({
        'user_sso': uid,
        'report_id': iid,
        'score_estimate': est},
        columns=['user_sso', 'report_id', 'score_estimate']
    )

    recommendation_df = recommendation_df.sort_values(
        by=['user_sso', 'score_estimate'],
        ascending=[True, False],
        inplace=False
    )

    recommendation_df['rank'] = recommendation_df.groupby(
        'user_sso'
    ).cumcount() + 1

    recommendation_df = recommendation_df.loc[
        recommendation_df['rank'] <= n] #.copy()  # type: pd.DataFrame'

    return recommendation_df


def generate_svd_recommendation_df() -> pd.DataFrame:
    # Prepare input DataFrame and algorithm
    score_df = genearte_score_df()
    svd_data = MyDataSet(score_df)
    #Try SVD
    algo = SVD()
    full_train_set = svd_data.build_full_trainset()
    test_set = full_train_set.build_anti_testset()
    # 5 fold validation
    score = cross_validate(algo, svd_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    # Fitting the SVD
    algo.fit(full_train_set)
    predictions = algo.test(test_set)
    # Then compute RMSE
    accuracy.rmse(predictions)
    # Generate recommendation DataFrame
    recommendation_df_svd = get_top_n(predictions, n=5)
    #print (recommendation_df)
    
    
    #Try the NMF
    nmf_cv = cross_validate(NMF(), svd_data, cv=5, n_jobs=5, verbose=False) 
    algo = NMF()
    full_train_set = svd_data.build_full_trainset()
    test_set = full_train_set.build_anti_testset()
    # 5 fold validation
    score = cross_validate(algo, svd_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    # Fitting the SVD
    algo.fit(full_train_set)
    predictions = algo.test(test_set)
    # Then compute RMSE
    accuracy.rmse(predictions)
    accuracy.mae(predictions)
    # Generate recommendation DataFrame
    recommendation_df_svd = get_top_n(predictions, n=5)
    #print (recommendation_df)
    
    
    
    #---------------------------------------------------
    # as per - https://bmanohar16.github.io/blog/recsys-evaluation-in-surprise
    knnbasic_cv = cross_validate(KNNBasic(), svd_data, cv=5, n_jobs=5, verbose=False)
    knnmeans_cv = cross_validate(KNNWithMeans(), svd_data, cv=5, n_jobs=5, verbose=False)
    knnz_cv = cross_validate(KNNWithZScore(), svd_data, cv=5, n_jobs=5, verbose=False)

    # Matrix Factorization Based Algorithms
    svd_cv = cross_validate(SVD(), svd_data, cv=5, n_jobs=5, verbose=False)
    svdpp_cv = cross_validate(SVDpp(),svd_data, cv=5, n_jobs=5, verbose=False)
    nmf_cv = cross_validate(NMF(), svd_data, cv=5, n_jobs=5, verbose=False) 
    
    #Other Collaborative Filtering Algorithms
    slope_cv = cross_validate(SlopeOne(), svd_data, cv=5, n_jobs=5, verbose=False)
    coclus_cv = cross_validate(CoClustering(), svd_data, cv=5, n_jobs=5, verbose=False)

#-------------==========================Print it out=====================================================
print('Algorithm\t RMSE\t\t MAE')
print()
print('KNN Basic', '\t', round(knnbasic_cv['test_rmse'].mean(), 4), '\t', round(knnbasic_cv['test_mae'].mean(), 4))
print('KNN Means', '\t', round(knnmeans_cv['test_rmse'].mean(), 4), '\t', round(knnmeans_cv['test_mae'].mean(), 4))
print('KNN ZScore', '\t', round(knnz_cv['test_rmse'].mean(), 4), '\t', round(knnz_cv['test_mae'].mean(), 4))
print()
print('SVD', '\t\t', round(svd_cv['test_rmse'].mean(), 4), '\t', round(svd_cv['test_mae'].mean(), 4))
print('SVDpp', '\t\t', round(svdpp_cv['test_rmse'].mean(), 4), '\t', round(svdpp_cv['test_mae'].mean(), 4))
print('NMF', '\t\t', round(nmf_cv['test_rmse'].mean(), 4), '\t', round(nmf_cv['test_mae'].mean(), 4))
print()
print('SlopeOne', '\t', round(slope_cv['test_rmse'].mean(), 4), '\t', round(slope_cv['test_mae'].mean(), 4))
print('CoClustering', '\t', round(coclus_cv['test_rmse'].mean(), 4), '\t', round(coclus_cv['test_mae'].mean(), 4))
print()

#-----------------------------------Plot it out----------------------------------------------------------------------------
x_algo = ['KNN Basic', 'KNN Means', 'KNN ZScore', 'SVD', 'SVDpp', 'NMF', 'SlopeOne', 'CoClustering']
all_algos_cv = [knnbasic_cv, knnmeans_cv, knnz_cv, svd_cv, svdpp_cv, nmf_cv, slope_cv, coclus_cv]
rmse_cv = [round(res['test_rmse'].mean(), 4) for res in all_algos_cv]
mae_cv = [round(res['test_mae'].mean(), 4) for res in all_algos_cv]
plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)
plt.title('Comparison of Algorithms on RMSE', loc='center', fontsize=15)
plt.plot(x_algo, rmse_cv, label='RMSE', color='darkgreen', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('RMSE Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')
plt.show()



x_algo = ['KNN Basic', 'KNN Means', 'KNN ZScore', 'SVD', 'SVDpp', 'NMF', 'SlopeOne', 'CoClustering']
all_algos_cv = [knnbasic_cv, knnmeans_cv, knnz_cv, svd_cv, svdpp_cv, nmf_cv, slope_cv, coclus_cv]
rmse_cv = [round(res['test_rmse'].mean(), 4) for res in all_algos_cv]
mae_cv = [round(res['test_mae'].mean(), 4) for res in all_algos_cv]
plt.figure(figsize=(20,5))
plt.subplot(1, 2, 2)
plt.title('Comparison of Algorithms on MAE', loc='center', fontsize=15)
plt.plot(x_algo, mae_cv, label='MAE', color='navy', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('MAE Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')
plt.show()

#================-----------------------------------------------------------------------------------------------------------
    
    
    
    return recommendation_df


if __name__ == '__main__':
    df = generate_svd_recommendation_df()
    print(f'SVD Recommendation df ==> {df.shape}')
    
latent_vect = algo.pu 
latent_item_v = algo.qi 
cross_validate(algo, svd_data, measures=[‘RMSE’, ‘MAE’], cv=5, verbose=True)

 

