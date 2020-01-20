#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created at 8/13/2019
__author__ = 212577071
Usage: 
"""

from typing import List

import numpy as np
import pandas as pd

from model.utility import get_static_user_data, get_static_visit_data
from model.utility import get_obiee_user_df, get_obiee_visit_df

MAXIMUM_REPORT_CANDIDATES = 5


def generate_user_bucket_recommendation_df(source='static'):
    source = 'static'
    if source == 'live':
        user_df = get_obiee_user_df()
        visit_df = get_obiee_visit_df()
    elif source == 'static':
        user_df = get_static_user_data()
        visit_df = get_static_visit_data()
    else:
        raise AttributeError(f'source should be either live or static while input is {source}')

    print(f'User DataFrame ==> {user_df.shape}')
    print(f'Visit DataFrame ==> {visit_df.shape}')

    visit_df = pd.merge(
        left=visit_df, right=user_df,
        left_on='user_sso', right_on='person_sso_id',
        how='left'
    )  # type: pd.DataFrame

    bucket_criteria = {
        'process': ['func_nm', 'family_nm'],  # when you are grouping by process, use these columns
        'bu': ['org_ind_foc_grp', 'org_bus_seg_id']  # when you are grouping by business, use these columns
    }
    recommendation = {}

    band_column = 'corp_bnd'
    for bucket_type, bucket_column in bucket_criteria.items():
        level1, level2 = bucket_column

        def get_agg_df(df: pd.DataFrame, default_columns: List[str]) -> pd.DataFrame:
            agg_list = bucket_column + [band_column, 'report_id']
            for column in default_columns:
                agg_list.remove(column)
            agg_df = df.groupby(agg_list).agg({'total_access': np.sum}).reset_index(inplace=False)
            for column in default_columns:
                agg_df[column] = 'DEFAULT'
            return agg_df

        full_criteria_w_band = get_agg_df(visit_df, default_columns=[])
        full_criteria_wo_band = get_agg_df(visit_df, default_columns=[band_column])
        l1_only_w_band = get_agg_df(visit_df, default_columns=[level2])
        l1_only_wo_band = get_agg_df(visit_df, default_columns=[level2, band_column])

        bucket_df = pd.concat(
            [full_criteria_w_band, full_criteria_wo_band,
             l1_only_w_band, l1_only_wo_band],
            axis=0, sort=True
        )
        bucket_df = bucket_df.loc[(bucket_df[level2].notnull()) & (bucket_df[band_column].notnull())]
        bucket_df = bucket_df.sort_values(
            by=bucket_column + [band_column, 'total_access'],
            ascending=False, inplace=False
        )  # type: pd.DataFrame
        bucket_df['rank_in_bucket'] = bucket_df.groupby(
            bucket_column + [band_column]
        ).cumcount() + 1

        bucket_recommendation_df = bucket_df.loc[
            bucket_df['rank_in_bucket'] <= MAXIMUM_REPORT_CANDIDATES
            ].copy()  # type: pd.DataFrame
        for ind, k in enumerate(bucket_column, 1):
            bucket_recommendation_df.rename(
                columns={k: f'level_{ind}'},
                inplace=True
            )
        recommendation[bucket_type] = bucket_recommendation_df

    return recommendation


def get_recommendation(recommendation_type: str, recommendation: dict, criteria: List[str]):
    if recommendation_type not in ('process', 'bu'):
        raise AttributeError(f'recommendation_type should be either process or bu, get {recommendation_type} instead')

    # TODO Adds logic for missing/unavaiable segment
    recommendation_df = recommendation[recommendation_type]
    recommendation_df = recommendation_df.loc[
        (recommendation_df['level_1'] == criteria[0]) &
        (recommendation_df['level_2'] == criteria[1]) &
        (recommendation_df['corp_bnd'] == criteria[2])
        ]

    print(f'Top {recommendation_df.shape[0]} popular report in the bucket:')
    for report in recommendation_df['report_id']:
        print(f'|--{report}')


if __name__ == '__main__':
    recommendation_dict = generate_user_bucket_recommendation_df()
