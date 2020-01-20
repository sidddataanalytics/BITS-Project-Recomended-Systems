#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created at 8/20/2019
__author__ = 212577071
Usage: 
"""

import pandas as pd


def user_validate_fds(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def user_validate_tableau(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def user_validate_mgm(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def user_validate_obiee(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[
        (df['user_sso'].str.len() == 9) &
        (df['user_sso'].str.isdigit())].copy()
    df['user_sso'] = df['user_sso'].astype('int')
    return df


def report_validate_fds(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def report_validate_tableau(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def report_validate_mgm(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def report_validate_obiee(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[
        (df['report_id'] != 0) &
        (df['report_url'].notnull())].copy()

    df = df.loc[
        (~(df['report_url'].str.startswith('/users/'))) &
        (~(df['report_url'].str.contains('Adhoc')))
        ].copy()
    return df
