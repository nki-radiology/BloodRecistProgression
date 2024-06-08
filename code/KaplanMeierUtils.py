import os
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import datetime
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times
from Utils import create_paths, read_pickled_data


def KM_median_OS_PFS(durations, event_observed, endpoint_event, group=''):
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed, label=endpoint_event)
    median_to_event_occurence = kmf.median_survival_time_
    print(f"{group} {endpoint_event}")
    print(f"median survival time: {median_to_event_occurence}")
    return kmf


def plot_KM_curves(KM_clfs, labels, folder_path):
    four_line_styles = ['-', '--', '.', '..']
    line_styles = [four_line_styles[i] for i in range(len(KM_clfs))]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    plt.ylim(0, 1)
    ax.set_xlabel('Timeline (days)', fontsize='large')
    ax.set_ylabel('Percentage of patients without progression', fontsize='large')

    for KM_clf, line_style in zip(KM_clfs, line_styles):
        KM_clf.plot(ax=ax, linestyle="-", ci_show=False)
    ax.grid()

    # plt.show()
    create_paths([folder_path])
    if len(labels) == 2 and len(KM_clfs) == 2:
        plot_name = f'KM_{labels[0]}_{labels[1]}.svg'
        plot_title = f'KM curves of {labels[0].upper()} and {labels[1].upper()}'
    else:
        plot_name = f'KM_{labels[0]}.svg'
        plot_title = f'KM curves of {labels[0].split("_")[0].upper()} grouped by ALP and CRP ranges'
    ax.set_title(plot_title)
    ax.legend().remove()
    plot_path = os.path.join(folder_path, plot_name)

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('svg', 'jpg'), dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def calculate_OS_days_from_start_date(row, start_date_col='treatment_date'):
    start_date = pd.to_datetime(row[start_date_col])

    if isinstance(row['death_date'], datetime):
        death_date = row['death_date']

    if ((str(row['death_date']) == "no") | (str(row['death_date']) == "alive")):
        OS_days_from_SoT = (row['last_check_date'] - start_date).days
        death_event = 0
    else:
        OS_days_from_SoT = (death_date - start_date).days
        death_event = 1

    return OS_days_from_SoT, death_event


def perform_KM_median_PFS_OS(blood_data):
    blood_df = blood_data.df
    blood_df = blood_df.drop_duplicates(subset=['anonym_id'])
    blood_df.rename(columns={'pfs': 'pfs_event'}, inplace=True)
    print(blood_data.df.shape)
    # add death event and os_days_from_SoT
    blood_df[['OS_days_from_SoT', 'death_event']] = blood_df.apply(lambda row: calculate_OS_days_from_start_date(row),
                                                                   axis=1, result_type="expand")
    kmf_os = KM_median_OS_PFS(blood_df['OS_days_from_SoT'], blood_df['death_event'], "OS")
    kmf_pfs = KM_median_OS_PFS(blood_df['pfs_days_from_SoT'], blood_df['pfs_event'], "PFS")
    plot_KM_curves(KM_clfs=[kmf_os, kmf_pfs],
                   labels=["OS", "PFS"],
                   folder_path=blood_data.output_exp_path)


def get_normal_blood_ranges(row, blood_value, condition, range):
    value = row[blood_value]
    sex = row['clin.sex']
    specific_range = range[sex]
    if 'less' in condition:
        is_normal = 1 if value < specific_range else 0

    else:
        is_normal = 1 if value > specific_range else 0

    return is_normal


def get_blood_KM_groups(row, col1, col2):
    if row[col1] == 0 and row[col2] == 0:
        group = 1
    elif row[col1] == 0 and row[col2] == 1:
        group = 2
    elif row[col1] == 1 and row[col2] == 0:
        group = 3
    elif row[col1] == 1 and row[col2] == 1:
        group = 4
    return group


def add_normal_abnormal_groups(blood_df, blood_markers_levels):
    for marker, marker_levels_dict in blood_markers_levels.items():
        blood_df[f'{marker}_is_normal'] = blood_df.apply(
            lambda row: get_normal_blood_ranges(row, blood_value=marker, condition="less", range=marker_levels_dict),
            axis=1)

    blood_df['KM_groups'] = blood_df.apply(
        lambda row: get_blood_KM_groups(row, col1=f'{list(blood_markers_levels.keys())[0]}_is_normal',
                                        col2=f'{list(blood_markers_levels.keys())[1]}_is_normal'), axis=1)

    return blood_df


def add_static_pfs_event_col(blood_df):
    blood_df = blood_df.drop_duplicates(subset=['anonym_id'])
    pfs_dict = dict(zip(blood_df['anonym_id'].to_list(), blood_df['pfs'].to_list()))
    return pfs_dict


def add_KM_durations_events_cols(blood_df, pre_cleaned_blood_df):
    blood_df[['OS_days_from_test_day', 'death_event']] = blood_df.apply(
        lambda row: calculate_OS_days_from_start_date(row, start_date_col='exam_date'),
        axis=1, result_type="expand")

    blood_df[['OS_days_from_SoT', 'death_event_2']] = blood_df.apply(
        lambda row: calculate_OS_days_from_start_date(row, start_date_col='treatment_date'),
        axis=1, result_type="expand")

    blood_df['pfs_days_from_test_day'] = blood_df['pfs_days_from_SoT'] - blood_df['start']
    blood_df.rename(columns={'pfs': 'pfs_tv_event'}, inplace=True)
    pfs_dict = add_static_pfs_event_col(pre_cleaned_blood_df)
    blood_df['pfs_event'] = blood_df['patient_ids'].map(pfs_dict)
    return blood_df


def apply_2y_cuttoff_KM(row, duration_col, event_col, cutoff_days=730):
    cutoff_duration = row[duration_col]
    event = row[event_col]
    if row[duration_col] > cutoff_days:
        cutoff_duration = cutoff_days
        event = 0

    return pd.Series([cutoff_duration, event])


def run_logrank_test_permuted_groups(labels, durations_col, events_col, grouped_blood_data):
    paired_groups = list(combinations(labels, 2))
    print("Log rank test:")
    for pair in paired_groups:
        print()
        print(pair)
        group_a_num = int(str(pair[0]).split(':')[0].split(" ")[-1])
        group_b_num = int(str(pair[1]).split(':')[0].split(" ")[-1])

        group_a_df = grouped_blood_data[grouped_blood_data['KM_groups'] == group_a_num]
        group_b_df = grouped_blood_data[grouped_blood_data['KM_groups'] == group_b_num]

        for duration, event in zip(durations_col, events_col):
            task = duration.split('_')[0].upper()
            print("\n" + task)
            results = logrank_test(group_a_df[duration], group_b_df[duration], event_observed_A=group_a_df[event],
                                   event_observed_B=group_b_df[event])
            results.print_summary()
            print(f"p-value: {results.p_value}")


def perform_KM_blood_test_levels(blood_data, inclusion_timepoint='late_pre_treat', add_2y_cutoff=False,
                                 only_tests_pre_pfs_event=True):
    pre_cleaned_blood_df = blood_data.df

    blood_data.clean_data()
    blood_df = blood_data.df
    # merge demographics
    demographics_df = blood_data.get_demographics_df()
    blood_df = pd.merge(blood_df, demographics_df, how='left', on='patient_ids')

    # ranges: CRP <8 normal , AF < 98 for females and <115 for males
    blood_markers_levels = {
        'ALP': {'F': 98, 'M': 115},
        'CRP': {'F': 8, 'M': 8}
    }
    grouped_blood_df = add_normal_abnormal_groups(blood_df, blood_markers_levels)
    grouped_blood_df = add_KM_durations_events_cols(grouped_blood_df, pre_cleaned_blood_df)

    # if only_six_month_interval:
    #     grouped_blood_df = grouped_blood_df.loc[(grouped_blood_df.start >= -92) & (grouped_blood_df.start <= 92), :]
    # grouped_blood_df = grouped_blood_df.loc[(grouped_blood_df.start >= 0) & (grouped_blood_df.start <= 92), :]
    # grouped_blood_df = grouped_blood_df.loc[grouped_blood_df.start >= 0, :] # from SoT

    # check inclusion criteria
    if inclusion_timepoint == 'late_pre_treat':  # pretreatment closest to treatment
        grouped_blood_df = grouped_blood_df.loc[grouped_blood_df.start <= 0]
        grouped_blood_df = grouped_blood_df.sort_values(by=['patient_ids', 'start'], ascending=False)
        grouped_blood_df = grouped_blood_df.drop_duplicates(subset=['patient_ids'])
        grouped_blood_df = grouped_blood_df.loc[grouped_blood_df.start >= -30]  # all satisfy this cond n=171

    elif inclusion_timepoint == 'early_treat':  # between 4th and 6th weeks after treatment, the closest to week 6
        grouped_blood_df = grouped_blood_df.loc[(grouped_blood_df.start >= 28) & (grouped_blood_df.start <= 42), :]
        grouped_blood_df = grouped_blood_df.sort_values(by=['patient_ids', 'start'], ascending=False)
        grouped_blood_df = grouped_blood_df.drop_duplicates(subset=['patient_ids'])  # 102

    elif inclusion_timepoint == 'late_early_treat':  # between 10 to 12 weeks, the closest to 12 weeks
        grouped_blood_df = grouped_blood_df.loc[(grouped_blood_df.start >= 70) & (grouped_blood_df.start <= 84), :]
        grouped_blood_df = grouped_blood_df.sort_values(by=['patient_ids', 'start'], ascending=False)
        grouped_blood_df = grouped_blood_df.drop_duplicates(subset=['patient_ids'])  # 102

    # check_blood_df=grouped_blood_df[grouped_blood_df['pfs_days_from_test_day']<0]
    # check_blood_df.to_excel('check_pfs_days.xlsx')

    if only_tests_pre_pfs_event:
        # make sure to include only the tests before the pfs event happened
        grouped_blood_df = grouped_blood_df[grouped_blood_df[
                                                'pfs_days_from_test_day'] > 0]  # 171 for BL, 93 for early_treat and 85 for late early treat

    for pfs_col in ['pfs_tv_event', 'pfs_event']:
        print(grouped_blood_df[pfs_col].value_counts())

    durations_col = ['pfs_days_from_test_day', 'OS_days_from_test_day']
    events_col = ['pfs_event', 'death_event']
    # grouped_blood_df.to_excel(os.path.join(blood_data.output_exp_path,'grouped_blood_df_updated.xlsx'),index=False)

    if add_2y_cutoff:
        cutoff_days = 730
        durations_col = ['pfs_days_from_test_day_2y_cutoff', 'OS_days_from_test_day_2y_cutoff']
        grouped_blood_df[['pfs_days_from_test_day_2y_cutoff', 'pfs_event']] = grouped_blood_df.apply(
            lambda row: apply_2y_cuttoff_KM(row, duration_col='pfs_days_from_test_day', event_col='pfs_event',
                                            cutoff_days=cutoff_days), axis=1, result_type='expand')
        grouped_blood_df[['OS_days_from_test_day_2y_cutoff', 'death_event']] = grouped_blood_df.apply(
            lambda row: apply_2y_cuttoff_KM(row, duration_col='OS_days_from_test_day', event_col='death_event',
                                            cutoff_days=cutoff_days), axis=1, result_type='expand')

        # print(grouped_blood_df[durations_col].isnull().sum())

        grouped_blood_df.to_excel(
            os.path.join(blood_data.output_exp_path, f'grouped_blood_df_2y_cutoff_{inclusion_timepoint}.xlsx'),
            index=False)

    print(
        f"number of included unique patients during {inclusion_timepoint}: {grouped_blood_df['patient_ids'].nunique()}")

    labels = [
        f'Group 1: Abnormal {list(blood_markers_levels.keys())[0]} and abnormal {list(blood_markers_levels.keys())[1]}',
        f'Group 2: Abnormal {list(blood_markers_levels.keys())[0]} and normal {list(blood_markers_levels.keys())[1]}',
        f'Group 3: Normal {list(blood_markers_levels.keys())[0]} and abnormal {list(blood_markers_levels.keys())[1]}',
        f'Group 4: Normal {list(blood_markers_levels.keys())[0]} and normal {list(blood_markers_levels.keys())[1]}']

    kmf_lists = {'PFS': [], 'OS': []}
    for i, label in enumerate(labels):
        b_data = grouped_blood_df[grouped_blood_df['KM_groups'] == i + 1]
        for duration, event in zip(durations_col, events_col):
            task = duration.split('_')[0].upper()
            print("\n" + task)
            kmf = KM_median_OS_PFS(b_data[duration], b_data[event], label.split(': ')[-1], group=label.split(': ')[0])
            kmf_lists[task].append(kmf)

    for task in kmf_lists.keys():
        plot_KM_curves(KM_clfs=kmf_lists[task],
                       labels=[f"{task}_blood_groups"],
                       folder_path=os.path.join(blood_data.output_exp_path, inclusion_timepoint))

    plt.clf()
    run_logrank_test_permuted_groups(labels, durations_col, events_col, grouped_blood_df)
