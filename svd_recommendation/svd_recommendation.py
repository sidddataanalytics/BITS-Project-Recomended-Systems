#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created at 8/20/2019
__author__ = 212577071
Usage: 
"""
import os
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

reports_id_csv = 'C:\\Sid Data\\BITS\\4th Sem\\Sidd code\\Sidd Code Base\\1.Data\\wrapperdb_public_reports.csv'
assert os.path.exists(reports_id_csv), 'OBIEE_VISIT_CSV does not exists'
df_reports_id = pd.read_csv(reports_id_csv, low_memory=False)
#return df[['person_sso_id', 'org_ind_foc_grp', 'org_bus_seg_id', 'func_nm', 'family_nm', 'corp_bnd']]


MAXIMUM_REPORT_CANDIDATES = 5


class MyDataSet(DatasetAutoFolds):
    def __init__(self, df, reader=Reader(line_format='user item rating', rating_scale=(1, 5))):
        super().__init__(reader=reader, df=df)
        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in zip(df['user_sso'], df['report_id'], df['score'])]
        a = self.raw_ratings
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
    print (score_df)
    print (svd_data.raw_ratings)
    #Try SVD
    algo_svd = SVD()
    full_train_set = svd_data.build_full_trainset()
    test_set = full_train_set.build_anti_testset()
    # 5 fold validation
    score = cross_validate(algo_svd, svd_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    # Fitting the SVD
    algo_svd.fit(full_train_set)
    predictions = algo_svd.test(test_set)
    # Then compute RMSE
    accuracy.rmse(predictions)
    # Generate recommendation DataFrame
    recommendation_df_svd = get_top_n(predictions, n=5)
    latent_usr_factor = algo_svd.pu 
    latent_item_factor = algo_svd.qi 
    user_bias = algo_svd.bu
    item_bias = algo_svd.bi
    recomendation_reportname_df_svd = pd.merge(recommendation_df_svd, df_reports_id, how = 'left', on= 'report_id')

    
    #Try SVD++
    algo_svdpp = SVDpp()
    full_train_set = svd_data.build_full_trainset()
    test_set = full_train_set.build_anti_testset()
    # 5 fold validation
    score = cross_validate(algo_svdpp, svd_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    # Fitting the SVD
    algo_svdpp.fit(full_train_set)
    predictions = algo_svdpp.test(test_set)
    # Then compute RMSE
    accuracy.rmse(predictions)
    # Generate recommendation DataFrame
    recommendation_df_svdpp = get_top_n(predictions, n=5)
    latent_usr_factor_pp = algo_svd.pu 
    latent_item_factor_pp = algo_svd.qi 
    user_bias_pp = algo_svd.bu
    item_bias_pp = algo_svd.bi
    recomendation_reportname_df_svdpp = pd.merge(recommendation_df_svdpp, df_reports_id, how = 'left', on= 'report_id')

      #Try SVD++ with more factors as default is 20
    algo_svdpp_mod = SVDpp(n_factors =50, n_epochs = 50)
    full_train_set = svd_data.build_full_trainset()
    test_set = full_train_set.build_anti_testset()
    # 5 fold validation
    score = cross_validate(algo_svdpp, svd_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    # Fitting the SVD
    algo_svdpp.fit(full_train_set)
    predictions = algo_svdpp.test(test_set)
    # Then compute RMSE
    accuracy.rmse(predictions)
    print (score)
    
    #print (recommendation_df)
    
    
    #Try the NMF
    #nmf_cv = cross_validate(NMF(), svd_data, cv=5, n_jobs=5, verbose=False) 
    algo_nmf = NMF()
    full_train_set = svd_data.build_full_trainset()
    test_set = full_train_set.build_anti_testset()
    # 5 fold validation
    score = cross_validate(algo_nmf, svd_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    # Fitting the SVD
    algo_nmf.fit(full_train_set)
    predictions = algo_nmf.test(test_set)
    # Then compute RMSE
    accuracy.rmse(predictions)
    accuracy.mae(predictions)
    # Generate recommendation DataFrame
    recommendation_df_nmf = get_top_n(predictions, n=5)
    #print (recommendation_df)
    latent_usr_factor_nmf = algo_svd.pu 
    latent_item_factor_nmf = algo_svd.qi 
    user_bias_nmf = algo_svd.bu
    item_bias_nmf = algo_svd.bi
    recomendation_reportname_df_mmf = pd.merge(recommendation_df_nmf, df_reports_id, how = 'left', on= 'report_id')
    sidd_recmidation = recomendation_reportname_df.loc[recomendation_reportname_df['user_sso'] == 212568816]
    
        #Try the NMF without default
    #nmf_cv = cross_validate(NMF(), svd_data, cv=5, n_jobs=5, verbose=False) 
    algo_nmf_mod = NMF(n_factors =50, n_epochs = 50)
    full_train_set = svd_data.build_full_trainset()
    test_set = full_train_set.build_anti_testset()
    # 5 fold validation
    score = cross_validate(algo_nmf, svd_data, measures=['RMSE', 'MAE'], cv=5, verbose=True, )
    # Fitting the SVD
    algo_nmf.fit(full_train_set)
    predictions = algo_nmf.test(test_set)
    # Then compute RMSE
    accuracy.rmse(predictions)
    accuracy.mae(predictions)
    # Generate recommendation DataFrame
    recommendation_df_nmf = get_top_n(predictions, n=5)
    #print (recommendation_df)
    latent_usr_factor_nmf = algo_svd.pu 
    latent_item_factor_nmf = algo_svd.qi 
    user_bias_nmf = algo_svd.bu
    item_bias_nmf = algo_svd.bi
    recomendation_reportname_df_mmf = pd.merge(recommendation_df_nmf, df_reports_id, how = 'left', on= 'report_id')
    sidd_recmidation = recomendation_reportname_df.loc[recomendation_reportname_df['user_sso'] == 212568816]
    
    
    #---------------------------------------------------
    # as per - https://bmanohar16.github.io/blog/recsys-evaluation-in-surprise
     # Matrix Factorization Based Algorithms
    svd_cv = cross_validate(algo_svd, svd_data, cv=5, n_jobs=5, verbose=False)
    svdpp_cv = cross_validate(algo_svdpp,svd_data, cv=5, n_jobs=5, verbose=False)
    nmf_cv = cross_validate(algo_nmf, svd_data, cv=5, n_jobs=5, verbose=False) 
    svdpp_cv_mod = cross_validate(algo_svdpp_mod,svd_data, cv=5, n_jobs=5, verbose=False)
    nmf_cv_mod = cross_validate(algo_nmf_mod, svd_data, cv=5, n_jobs=5, verbose=False) 
#-------------==========================Print it out=====================================================
print('Algorithm\t\t RMSE\t\t MAE')
print()
print('SVD      ', '\t\t', round(svd_cv['test_rmse'].mean(), 4), '\t', round(svd_cv['test_mae'].mean(), 4))
print('SVDpp    ', '\t\t', round(svdpp_cv['test_rmse'].mean(), 4), '\t', round(svdpp_cv['test_mae'].mean(), 4))
print('SVDpp_mod', '\t\t', round(svdpp_cv_mod['test_rmse'].mean(), 4), '\t', round(svdpp_cv_mod['test_mae'].mean(), 4))
print('NMF      ', '\t\t', round(nmf_cv['test_rmse'].mean(), 4), '\t', round(nmf_cv['test_mae'].mean(), 4))
print('NMF_mod  ', '\t\t', round(nmf_cv_mod['test_rmse'].mean(), 4), '\t', round(nmf_cv_mod['test_mae'].mean(), 4))
print()
print()


print('Algorithm\t RMSE\t\t MAE')
print()
print('SVD', '\t\t', round(svd_cv['test_time'].mean(), 4), '\t', round(svd_cv['fit_time'].mean(), 4))
print('SVDpp', '\t\t', round(svdpp_cv['test_time'].mean(), 4), '\t', round(svdpp_cv['fit_time'].mean(), 4))
print('NMF', '\t\t', round(nmf_cv['test_time'].mean(), 4), '\t', round(nmf_cv['fit_time'].mean(), 4))
print()
print()

#-----------------------------------Plot it out----------------------------------------------------------------------------
x_algo = ['SVD', 'SVDpp','SVDpp_mod', 'NMF', 'NMF_mod']
all_algos_cv = [svd_cv, svdpp_cv, svdpp_cv_mod, nmf_cv, nmf_cv_mod]
rmse_cv = [round(res['test_rmse'].mean(), 4) for res in all_algos_cv]
mae_cv = [round(res['test_mae'].mean(), 4) for res in all_algos_cv]
plt.figure(figsize=(40,10))

plt.subplot(1, 2, 1)
plt.title('Comparison of Algorithms on RMSE', loc='center', fontsize=15)
plt.plot(x_algo, rmse_cv, label='RMSE', color='darkgreen', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('RMSE Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')
plt.show()



x_algo = ['SVD', 'SVDpp', 'SVDpp_mod', 'NMF', 'NMF_mod']
all_algos_cv = [svd_cv, svdpp_cv, svdpp_cv_mod, nmf_cv, nmf_cv_mod]
rmse_cv = [round(res['test_rmse'].mean(), 4) for res in all_algos_cv]
mae_cv = [round(res['test_mae'].mean(), 4) for res in all_algos_cv]
plt.figure(figsize=(40,10))
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

 

