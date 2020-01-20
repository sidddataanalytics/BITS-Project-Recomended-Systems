#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created at 8/20/2019
__author__ = 212577071
Usage: 
"""

import numpy as np
import pandas as pd

from model.utility import get_static_visit_data, get_obiee_visit_df

source = 'static'
def genearte_score_df(source='static') -> pd.DataFrame:
    if source == 'static':
        visit_df = get_static_visit_data()
    elif source == 'live':
        visit_df = get_obiee_visit_df()
    else:
        raise NotImplementedError(f'{source} is a invalid source')
    score_df = visit_df[['user_sso', 'report_id', 'total_access']].copy()
    cut_off = np.percentile(score_df['total_access'], 99)
    

    def min_max_scaler(_, min_score=1, max_score=5):
        if _ >= cut_off:
            return max_score
        else:
            return (_ - 1) * (max_score - min_score) / (cut_off - 1) + min_score

    score_df['score'] = score_df['total_access'].apply(min_max_scaler)
    score_df['total_access'].describe()
    score_df['score'].describe()
    

    return score_df[['user_sso', 'report_id', 'score']]
