#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created at 8/13/2019
__author__ = 212577071
Usage: 
"""

import os
import pandas as pd

from model.utility.data_validation import user_validate_obiee, report_validate_obiee

#OBIEE_VISIT_CSV = '../../1.Data/2019-OBIEE-Visit-w-report.csv'
OBIEE_VISIT_CSV = 'C:\\Sid Data\\BITS\\4th Sem\\Sidd code\\Sidd Code Base\\1.Data\\2019-OBIEE-Visit-w-report.csv'
EMPLOY_ASSIGNMENTS_CSV = 'C:\\Sid Data\\BITS\\4th Sem\\Sidd code\\Sidd Code Base\\1.Data\\xx_hr_employ_assignments.csv'
REPORT_CONTENT_CSV = 'C:\\Sid Data\\BITS\\4th Sem\\Sidd code\\Sidd Code Base\\1.Data\\Users_report__details_crosstab.csv'

assert os.path.exists(OBIEE_VISIT_CSV), 'OBIEE_VISIT_CSV does not exists'
assert os.path.exists(EMPLOY_ASSIGNMENTS_CSV), 'EMPLOY_ASSIGNMENTS_CSV does not exists'
assert os.path.exists(REPORT_CONTENT_CSV), 'REPORT_CONTENT_CSV does not exists'


def get_static_user_data():
    df = pd.read_csv(EMPLOY_ASSIGNMENTS_CSV, low_memory=False)
    return df[['person_sso_id', 'org_ind_foc_grp', 'org_bus_seg_id', 'func_nm', 'family_nm', 'corp_bnd']]


def get_static_visit_data():
    df = pd.read_csv(OBIEE_VISIT_CSV, low_memory=False)
    print(f'Raw dataframe ==> {df.shape}')
    # Filter out observation with invalid SSO
    # SSO should be a 9-digit sequence
    df = user_validate_obiee(df)
    print(f'DataFrame after sso filter ==> {df.shape}')

    # Filter out invalid report
    df = report_validate_obiee(df)
    print(f'DataFrame after report filter ==> {df.shape}')
    df['total_access'].describe()
    return df


def get_static_report_data() -> pd.DataFrame:
    df = pd.read_csv(REPORT_CONTENT_CSV)
    df.columns = ['Name', 'SSO', 'Biz', 'Band', 'Country', 'Product', 'Report_Name', 'Rating']

    for col in ['Product', 'Report_Name']:
        # Standardize the Product and Report name by UPPERCASING and remove blanks
        df[col] = df[col].str.upper()
        df[col] = df[col].str.strip()

    core_product = ['FDS', 'FNOBIEE', 'FNTABLEAU']

    core_report_clean_df = df.loc[
        # Retain only the FDS, FNOBIEE & TABLEAU products
        (df['Product'].isin(core_product)) &
        # Remove any reports with 'Prompt' or 'test' in it's name
        ~((df['Report_Name'].str.contains('PROMPT')) |
          (df['Report_Name'].str.contains('TEST')))
        ]

    # Remove duplication by only keeping the report related information
    core_report_df = core_report_clean_df[
        ['Product', 'Report_Name']
    ].drop_duplicates(inplace=False).reset_index(
        drop=True,
        inplace=False
    )

    print(f'Original DataFrame ==> {df.shape}')
    print(f'Cleaned DataFrame ==> {core_report_clean_df.shape}')
    print(f'De-duped DataFrame ==> {core_report_df.shape}')

    return core_report_df


if __name__ == '__main__':
    get_static_report_data()
