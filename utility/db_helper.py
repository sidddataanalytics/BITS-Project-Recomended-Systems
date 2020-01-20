#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created at 8/21/2019
__author__ = 212577071
Usage: 
"""
import pandas as pd

from db_config import connection_session


def get_data(query, columns, engine, params=None) -> pd.DataFrame:
    with connection_session(engine) as f:
        df = pd.read_sql(sql=query, con=f, columns=columns, params=params)
    return df
