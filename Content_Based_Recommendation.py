# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created at 8/27/2019
__author__ = 212577071
Usage: The code is originally written by Siddhartha
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model.utility import get_static_report_data

MAXIMUM_REPORT_CANDIDATES = 5


##def genearte_content_similarity_recommendation():
    #Get the data 
    report_df = get_static_report_data()

    # Report to report similarity
    report_df['product_info'] = report_df['Product'] + ' ' + report_df['Report_Name']

    # Generate TF-IDF vector
    cv = CountVectorizer(stop_words='english')
    report_vector = cv.fit_transform(report_df['product_info'])

    # Calculate pairwise consine similarity
    report_cosine_similarity = cosine_similarity(report_vector)

    # Convert similarity from ndarray to Pandas DataFrame with top candidates
    report_id = []
    candidate_id = []
    for ind in range(len(report_cosine_similarity)):
        similarity = report_cosine_similarity[ind]  # type: np.ndarray
        top_candidates = similarity.argsort()[-MAXIMUM_REPORT_CANDIDATES - 1:-1]
        report_id.extend([ind] * MAXIMUM_REPORT_CANDIDATES)
        candidate_id.extend(top_candidates)

    candidate_df = pd.DataFrame({
        'report_id': report_id,
        'candidate_id': candidate_id
    })

    # Fetch similarity and report name
    candidate_df['similarity'] = candidate_df.apply(
        lambda _: report_cosine_similarity[_['report_id'], _['candidate_id']],
        axis=1
    )
    candidate_df['report_name'] = candidate_df['report_id'].apply(
        lambda _: report_df.loc[_, 'product_info']
    )
    candidate_df['candidate_name'] = candidate_df['candidate_id'].apply(
        lambda _: report_df.loc[_, 'product_info']
    )

    # Adding ranks within each report
    candidate_df = candidate_df.sort_values(
        by=['report_id', 'similarity'],
        ascending=[True, False],
        inplace=False
    )
    candidate_df['rank'] = candidate_df.groupby(by='report_id').cumcount() + 1

#return candidate_df