#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created at 8/21/2019
__author__ = 212577071
Usage: 
"""
from typing import List

import pandas as pd
from sqlalchemy import text, bindparam

from db_config import FDL_GP_PROD_ENGINE
from .db_helper import get_data
from .get_visit_data import get_obiee_visit_df


def get_employee_df(sso_list: List[str]) -> pd.DataFrame:
    # noinspection SqlResolve
    query = """
            SELECT
                person_sso_id,
                bus_grp_nm,
                org_ind_foc_grp,
                org_bus_seg_id,
                func_nm,
                family_nm,
                corp_bnd
            FROM
                analytics_view.xx_hr_employee_assignments_v
            WHERE
                  source_system_name = 'HR3'
              AND curr_flg = 'Y'
              AND person_sso_id IN :sso
            """
    query = text(query).bindparams(bindparam('sso', expanding=True))
    columns = [
        'user_sso',
        'bus_grp_nm', 'org_ind_foc_grp', 'org_bus_seg_id',
        'func_nm', 'family_nm', 'corp_bnd'
    ]
    df = get_data(query=query, columns=columns, engine=FDL_GP_PROD_ENGINE, params={'sso': sso_list})
    return df


def get_obiee_user_df() -> pd.DataFrame:
    obiee_visit_df = get_obiee_visit_df()
    sso_list = obiee_visit_df['user_sso'].drop_duplicates().values.tolist()
    obiee_user_df = get_employee_df(sso_list)
    return obiee_user_df
