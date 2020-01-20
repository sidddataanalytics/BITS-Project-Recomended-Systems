#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created at 8/20/2019
__author__ = 212577071
Usage: 
"""

import pandas as pd

from db_config import FDL_WRAPPER_DEV_ENGINE
from .db_helper import get_data
from .data_validation import user_validate_obiee, report_validate_obiee


def get_obiee_visit_df() -> pd.DataFrame:
    query = """
            SELECT
                user_sso,
                report_id,
                report_url,
                sum(access_count) AS total_access
            FROM
                fdl_metrics_reports_usage 
            WHERE
                  extract(YEAR FROM accessed_date) >= 2019
              AND product_type = 2
              AND report_id <> 34152
              AND report_id <> 0
            GROUP BY user_sso, report_id, report_url
            ORDER BY user_sso, report_id;
            """

    columns = ['user_sso', 'report_id', 'report_url', 'total_access']
    df = get_data(query, columns, FDL_WRAPPER_DEV_ENGINE)
    df = user_validate_obiee(df)
    df = report_validate_obiee(df)
    return df
