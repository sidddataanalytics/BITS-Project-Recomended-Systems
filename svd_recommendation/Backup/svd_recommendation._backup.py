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
    recommendation_df = get_top_n(predictions, n=5)
    print (recommendation_df)
    
    #---------------------------------------------------
    # as per - https://bmanohar16.github.io/blog/recsys-evaluation-in-surprise
    
    
    
    return recommendation_df


if __name__ == '__main__':
    df = generate_svd_recommendation_df()
    print(f'SVD Recommendation df ==> {df.shape}')
    
latent_vect = algo.pu 
latent_item_v = algo.qi 
cross_validate(algo, svd_data, measures=[‘RMSE’, ‘MAE’], cv=5, verbose=True)

 

