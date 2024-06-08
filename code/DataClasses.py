import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import Utils


class BloodData(object):
    def __init__(self, pred_future_timepoint=True, include_only_alive_patients=False, is_predictive=True,
                 os_from_pfs=False, keep_test_date=False, is_iRECIST=False, **params):
        self.output_exp_path = params['output_exp_path']
        self.data_folder_path = params['data_folder_path']
        self.all_data_stats_path = os.path.join(params['data_folder_path'],'clinical_dataset.xlsx')

        self.blood_data_type = params['blood_data_type']
        self.is_iRECIST = is_iRECIST
        if self.is_iRECIST:
            self.iRECIST_labels_path = os.path.join(self.data_folder_path, params['iRECIST_PATH'])

        self.pred_future_timepoint = pred_future_timepoint
        self.include_only_alive_patients = include_only_alive_patients
        self.is_predictive = is_predictive
        self.os_from_pfs = os_from_pfs
        self.keep_test_date = keep_test_date

        # label columns
        self.pfs_endpoints = params['pfs_endpoints_cols']
        self.os_endpoints = params['os_endpoints_cols']
        self.current_progression = params['current_progression_col']
        self.predictive_endpoints = self.pfs_endpoints + self.os_endpoints + [self.current_progression]
        self.other_endpoints = params['association_endpoints_cols']

        self.pred_days_from_SoT = params['pred_days_from_SoT']
        self.blood_markers_files_names = params['blood_markers_files_names_dict']
        # first blood feature in the excel file
        self.first_feature_col_names = params['first_feature_col_names_dict']
        self.features_names_abv_mapping = params['features_names_abv_mapping']

        # required dataset columns
        self.id_col = params['id_col']
        self.exam_date_col = params['exam_date_col']
        self.id_patients_col = params['updated_id_col']
        self.pfs_event_col = params['pfs_event_col']
        self.os_event_col = params['os_event_col']
        self.prog_due_to_death_col = params['prog_due_to_death_col']
        self.death_date_col = params['death_date_col']
        self.last_check_date_col = params['last_check_date_col']
        self.treatment_date_col = params['treatment_date_col']
        self.progression_date_col = params['progression_date_col']
        self.pfs_days_from_SoT_col = params['pfs_days_from_SoT_col']
        self.diff_death_from_prog_col = params['diff_death_from_prog_col']
        self.os_days_from_exam_date_col = params['os_days_from_exam_date_col']
        self.lab_test_from_SoT_col = params['lab_test_from_SoT_col']
        self.age_col = params['age_col']
        self.sex_col = params['sex_col']
        self.exam_to_last_check_days_col = params['exam_to_last_check_days_col']

        self.set_targets()

        # create_paths([self.output_exp_path])

    def set_targets(self):
        if self.is_predictive:
            if self.pred_future_timepoint:
                self.targets = self.pfs_endpoints
                # if self.os_from_pfs:
                #     self.targets = self.os_endpoints
            else:
                self.targets = [self.current_progression]
        else:
            self.targets = self.other_endpoints

    def load_data(self):
        path = os.path.join(self.data_folder_path, self.blood_markers_files_names[self.blood_data_type])
        self.first_blood_feature = self.first_feature_col_names[self.blood_data_type]
        self.df = pd.read_excel(path)
        self.blood_data_start_idx = (list(self.df)).index(self.first_blood_feature)

        if self.is_iRECIST:
            self.create_iRECIST_patients_lists()

    def create_iRECIST_patients_lists(self):
        # Not used in the current version of the paper
        iRECIST_df = pd.read_excel(self.iRECIST_labels_path)
        true_prog_col = "true progression"
        self.unconfirmed_iRECIST_patients = iRECIST_df[iRECIST_df[true_prog_col].isnull()][self.id_col].unique()
        self.pseudo_prog_patients = iRECIST_df[iRECIST_df[true_prog_col] == 'no'][self.id_col].unique()

    def replace_undesired_values(self):
        self.df.replace({
            '<1': '0', '<0.3': '0', '<0.1': '0', '<2': '0', '<3': '0',
            '>120': '130', '<5': '0', '<0.5': '0', '<Hem': np.NaN,
            '{<mg}': np.NaN, '{<gm}': np.NaN, '{<mvg}': np.NaN, '{<aggr}': np.NaN,
            '{<Onv-M}': np.NaN, '{<mng}': np.NaN, '{<Hem}': np.NaN, '{<Lip}': np.NaN,
            '{<Aanw}': np.NaN
        }, inplace=True)
        if "TUMOR_MARKERS" in self.blood_data_type:
            self.df['NSE'].replace({'Hemolytisch': np.NaN, 'Hemolyt.': np.NaN}, inplace=True)
            self.df['SCC'].replace({'Materiaal niet geschikt.': np.NaN, 'Geen materiaal ontvangen': np.NaN},
                                   inplace=True)

    def get_markers_names_dict(self):
        abv_features_dict = self.features_names_abv_mapping
        return abv_features_dict

    def set_blood_markers(self):
        self.blood_markers = list(self.df.columns[self.blood_data_start_idx:])

    def get_processed_blood_markers_start_idx(self):
        df_columns = list(self.df.columns)
        start_col_idx = df_columns.index(self.blood_markers[0])
        return start_col_idx

    def set_selected_blood_markers(self, start_col_idx):
        self.selected_blood_markers = list(self.df.columns[start_col_idx:])

    def get_num_tests_per_patient(self, df):
        print(f"\nall unique patients: {df[self.id_patients_col].nunique()}")
        num_blood_tests = df.groupby(self.id_patients_col).size()
        num_blood_tests_df = num_blood_tests.rename_axis('unique_values').reset_index(name='counts')
        print(
            f"\nDataset size: {len(df)}\nNumber of tests per patient: mean: {num_blood_tests_df['counts'].mean()}, median: {num_blood_tests_df['counts'].median()}")

    def get_demographics_df(self):
        demographics_df = pd.read_excel(self.all_data_stats_path)
        demographics_df = demographics_df[['patient', self.age_col, self.sex_col]]
        demographics_df.rename(columns={'patient': self.id_patients_col}, inplace=True)

        return demographics_df

    def calc_patients_stats(self, df=None):
        '''
        Calculates the cohort statistics
        '''
        print("\nPatients status...")
        if df is None:
            df = self.selected_df
        if self.age_col not in df.columns:
            demographics_df = self.get_demographics_df()
            p_df = df[[self.id_patients_col]].drop_duplicates()
            stats_data = pd.merge(p_df, demographics_df, how="left", on=self.id_patients_col)

        else:
            stats_data = df.drop_duplicates(subset=self.id_patients_col)
        print(
            f"Included patients: {stats_data[self.id_patients_col].nunique()} between ages of {stats_data[self.age_col].min()} and {stats_data[self.age_col].max()}\nMedian age: {stats_data[self.age_col].median()}, Average age: {stats_data[self.age_col].mean()}")

        print("Percentages of sex:")
        print(stats_data[self.sex_col].value_counts(normalize=True))

        self.get_num_tests_per_patient(df)

    def create_intervals_targets(self, blood_data):
        '''
        Define start and stop points for time-varying analysis regression and calculate the ground truth labels for the predictive endpoints

        '''

        if self.is_iRECIST:
            blood_data = blood_data[~blood_data[self.id_col].isin(self.unconfirmed_iRECIST_patients)]

        cols_to_calc = [self.exam_to_last_check_days_col, self.os_event_col] + self.os_endpoints[
                                                                               ::-1] + self.pfs_endpoints[::-1] + [
                           self.current_progression, 'interval', 'stop', 'stop_pfs', 'start']

        for col_to_calc in cols_to_calc:
            blood_data.insert(1, col_to_calc, list(np.zeros(len(blood_data))))

        blood_data.rename(columns={self.id_col: self.id_patients_col}, inplace=True)

        blood_data[self.death_date_col] = blood_data[self.death_date_col].replace({'no': np.NaN, 'alive': np.NaN})
        blood_data[[self.exam_date_col, self.treatment_date_col, self.death_date_col, self.last_check_date_col]] = \
            blood_data[
                [self.exam_date_col, self.treatment_date_col, self.death_date_col, self.last_check_date_col]].apply(
                pd.to_datetime,
                errors='raise')
        blood_data[self.death_date_col] = blood_data[self.death_date_col].fillna('no')

        # remove the last row if the patient's death date is the same as the last exam
        blood_data = blood_data[blood_data[self.death_date_col] != blood_data[self.exam_date_col]]

        # remove the tests that are acquired after the date of last check
        blood_data = blood_data[blood_data[self.exam_date_col] <= blood_data[self.last_check_date_col]]

        patient_ids = blood_data[self.id_patients_col].unique()

        updated_blood_dfs = []
        for k, pat in enumerate(patient_ids):

            pat_df = blood_data[blood_data[self.id_patients_col] == pat]
            pat_df = pat_df.sort_values(by=[self.exam_date_col])
            pat_df = pat_df.reset_index(drop=True)

            last_check = pat_df.loc[0, self.last_check_date_col]
            start_of_treatment = pat_df.loc[0, self.treatment_date_col]
            date_of_death = pat_df.loc[0, self.death_date_col]

            for exam in range(len(pat_df)):

                # everything needs to be referenced back to SoT date
                diff_between_exam_and_SoT = (pat_df.loc[exam, self.exam_date_col] - start_of_treatment).days

                # Let's first define START. Start is the date of the exam relative to the SoT
                #     If SoT happened after the exam, then the START will be negative

                if exam == 0:
                    pat_df.loc[exam, 'start'] = diff_between_exam_and_SoT
                else:  # the start of the next period is the stop of the last period
                    pat_df.loc[exam, 'start'] = pat_df.loc[exam - 1, 'stop']

                # Let's define STOP

                # if a patient only has one exam or if we reach the end of the exam list
                # then the last period is gonna be counted until death/last checkup
                if (len(pat_df) == 1) or (len(pat_df) == exam + 1):
                    # if the patient is dead. The period is from the time of the exam to
                    if date_of_death != 'no':
                        diff_to_previous = (date_of_death - pat_df.loc[exam, self.exam_date_col])
                        pat_df.loc[exam, self.os_event_col] = 1
                    else:  # if the patient is not dead, we use the last check up as the final date
                        diff_to_previous = (last_check - pat_df.loc[exam, self.exam_date_col]).days
                        pat_df.loc[exam, self.os_event_col] = 0

                else:  # if there is more than one exam
                    # Let's compute the difference in days between the exams
                    # difference (in days) to next exam
                    diff_to_previous = (
                            pat_df.loc[exam + 1, self.exam_date_col] - pat_df.loc[exam, self.exam_date_col]).days
                    pat_df.loc[exam, self.os_event_col] = 0

                # Update the stop
                if (len(pat_df) != 1) and (exam != 0):  # If it is not the first exam and there is more than one exam:
                    if isinstance(diff_to_previous, int):
                        diff_to_previous = diff_to_previous + pat_df.loc[exam - 1, 'stop']
                    else:
                        diff_to_previous = diff_to_previous.days + pat_df.loc[exam - 1, 'stop']

                # Let's adjust for SoT
                if exam == 0:
                    if isinstance(diff_to_previous, int):
                        diff_to_previous = diff_to_previous + pat_df.loc[exam, 'start']
                    else:
                        diff_to_previous = diff_to_previous.days + pat_df.loc[exam, 'start']

                pat_df.loc[exam, 'stop'] = diff_to_previous

                # adjust for iRECIST
                if self.is_iRECIST:
                    if pat in self.pseudo_prog_patients:
                        pat_df.loc[exam, self.pfs_event_col] = 0

                pfs_event = pat_df.loc[exam, self.pfs_event_col]
                # if progression happened: check if PFS from SoT is calculated between last check and SoT
                if pfs_event == 0:
                    pat_df.loc[exam, self.pfs_days_from_SoT_col] = (last_check - start_of_treatment).days

                pfs_event_cond = pfs_event == 1
                # If the exams go beyond the date of PFS, we just repeat the stop values and we'll eliminate those later
                if ((pat_df.loc[exam, 'stop'] >= pat_df.loc[exam, self.pfs_days_from_SoT_col]) & (pfs_event_cond)):
                    pat_df.loc[exam, 'stop_pfs'] = pat_df.loc[exam, self.pfs_days_from_SoT_col]
                    pat_df.loc[exam, self.pfs_event_col] = 1
                else:
                    pat_df.loc[exam, 'stop_pfs'] = pat_df.loc[exam, "stop"]
                    pat_df.loc[exam, self.pfs_event_col] = 0

                # If the patient reached PFS under X months, then we set pfs of X months to 1
                for duration in list(self.pred_days_from_SoT.keys()):
                    if ((pat_df.loc[exam, 'start'] + self.pred_days_from_SoT[duration] > pat_df.loc[
                        exam, self.pfs_days_from_SoT_col]) & (pfs_event_cond)):
                        pat_df.loc[exam, f'pfs_{duration}'] = 1

                    death_occured_cond = pat_df.loc[exam, self.death_date_col] != 'no'

                    pat_df.loc[exam, f'os_{duration}'] = 1 if (
                            (pat_df.loc[exam, self.os_days_from_exam_date_col] < self.pred_days_from_SoT[duration]) & (
                        death_occured_cond)) else 0

                pat_df.loc[exam, self.exam_to_last_check_days_col] = (
                        last_check - pat_df.loc[exam, self.exam_date_col]).days
            updated_blood_dfs.append(pat_df)

        self.df = pd.concat(updated_blood_dfs)
        self.df.reset_index(drop=True, inplace=True)
        self.df[self.current_progression] = np.where(self.df['start'] >= self.df[self.pfs_days_from_SoT_col], 1, 0)

        # add blood tests acquisition intervals stats
        self.df['interval'] = self.df['stop'] - self.df['start']
        # self.df['exam_to_last_check']= self.df['last_check']-self.df[se]
        print("Blood tests acquisition intervals (in days): ")
        print(f"mean: {self.df['interval'].mean()}, median: {self.df['interval'].median()}")

    def select_included_exams(self):
        '''
        Selects included longitudinal data
        '''
        if self.is_predictive:
            if self.pred_future_timepoint:
                self.selected_df = self.df[(self.df.start >= -92) & (self.df.start <= 92)]
            else:
                self.selected_df = self.df[(self.df.start >= 0) & (self.df.start <= self.df.pfs_days_from_SoT + 92)]
        else:
            self.selected_df = self.df

    def exclude_progressed_due_to_death_patients(self):
        self.selected_df = self.selected_df[~self.selected_df[self.prog_due_to_death_col]]

    def check_prevalent_blood_markers(self, start_col_idx):
        print("\nCounts of missing values:")
        missing_vals = self.df[self.df.columns[start_col_idx:]].isnull().sum().to_dict()
        sorted_missing_vals = {k: v for k, v in sorted(missing_vals.items(), key=lambda item: item[1])}
        print(sorted_missing_vals)
        most_common_markers = list(sorted_missing_vals.keys())
        print(
            f"\nMost 3 prevalent markers: {most_common_markers[0]}, {most_common_markers[1]}, {most_common_markers[2]}")

    def clean_data(self):
        self.set_blood_markers()
        self.df = self.df.sort_values(by=[self.id_col, self.exam_date_col])
        self.df = self.df.drop_duplicates(subset=self.blood_markers + [self.id_col, self.exam_date_col],
                                          ignore_index=True)
        missing_all_path = os.path.join(*[self.data_folder_path, 'MissingValues', 'all markers with undesired values'])
        missing_filtered_path = os.path.join(
            *[self.data_folder_path, 'MissingValues', 'all markers without undesired values'])

        Utils.create_paths([missing_all_path, missing_filtered_path])
        # self.df.to_excel(os.path.join(missing_all_path, f'missing_vals_all_{self.blood_data_type}.xlsx'), index=False)

        self.replace_undesired_values()
        # self.df.to_excel(os.path.join(missing_filtered_path, f'missing_vals_all_{self.blood_data_type}.xlsx'),index=False)

        for k in self.blood_markers:
            self.df = self.df.astype({k: float}, errors='raise')

        self.create_intervals_targets(self.df)

        # delete not required columns
        not_req_cols = [self.os_days_from_exam_date_col, self.progression_date_col, 'interval']

        if not self.keep_test_date:
            not_req_cols.extend(
                [self.exam_date_col, self.death_date_col, self.last_check_date_col, self.treatment_date_col])

        self.df = self.df.drop(columns=not_req_cols)
        # drop unimportant/underrepresetned features and/or exams
        self.df = self.drop_features(self.df)
        start_col_idx = self.get_processed_blood_markers_start_idx()
        self.df = self.drop_exams(self.df, start_col_idx)
        self.df.reset_index(drop=True, inplace=True)
        # Rename features
        self.df.rename(columns=self.features_names_abv_mapping, inplace=True)
        self.set_selected_blood_markers(start_col_idx)
        # Check prevalent blood markers
        self.check_prevalent_blood_markers(start_col_idx)

        # Get overall tests stats
        self.get_num_tests_per_patient(self.df)

        # Impute the missing values
        self.impute_features(start_col_idx)

    def impute_features(self, start_col_idx):
        print('\nImputing missing values...')
        blood_features = self.df.iloc[:, start_col_idx:]
        min_vec = np.array(blood_features.describe().loc[['min']]).flatten()
        max_vec = np.array(blood_features.describe().loc[['max']]).flatten()
        imp = IterativeImputer(max_iter=100, random_state=0, min_value=min_vec, max_value=max_vec)
        imp.fit(blood_features)
        imputed = imp.transform(blood_features)
        self.df.iloc[:, start_col_idx:] = imputed
        # drop columns if they are all zeros after imputation
        self.df = self.df.loc[:, (self.df != 0).any(axis=0)]
        print()

    def drop_features(self, blood_data, threshold=0.25):
        print('\nRemoving underrepresented features...')
        # See if there are enough values for each feature (more than 1/4), if not drop it
        minimun_nb_col = int(threshold * blood_data.shape[0])
        return blood_data.dropna(axis='columns', thresh=minimun_nb_col)

    def drop_exams(self, blood_data, start_col_idx, threshold=0.25):
        print('\nRemoving underrepresented exams...')
        blood_data.reset_index(drop=True, inplace=True)
        rows = np.array(blood_data.iloc[:, start_col_idx:].count(axis=1)) * 100 / \
               blood_data.iloc[:, start_col_idx:].shape[1]
        rows2 = rows > (threshold * 100)
        rows2_idx = np.squeeze(np.argwhere(rows2)).tolist()
        selected_df = blood_data.loc[rows2_idx]
        return selected_df

    def drop_patients(self, blood_data, patient_ids, start_col_idx, threshold=0.25):  # not used
        print('\nRemoving underrepresented patients...')
        # See if there are enough exams with values for each patient, if not drop the patient
        nblood_data = pd.DataFrame()
        for k, pat in enumerate(np.unique(patient_ids)):
            # get this patient's exams
            pat_df = blood_data[blood_data[self.id_patients_col] == pat]
            pat_df = pat_df.reset_index(drop=True)
            # count
            rows = np.array(pat_df.iloc[:, start_col_idx:].count(axis=1)) * 100 / \
                   pat_df.iloc[:, start_col_idx:].shape[1]

            # if more than half of the exams have at least half of the values.
            if (rows > threshold * 100).sum() * 100 / len(rows) > 50:
                nblood_data = nblood_data.append(pat_df)

        return nblood_data

    def drop_unrequired_columns(self):
        not_req_cols = [self.pfs_days_from_SoT_col, self.diff_death_from_prog_col]
        if self.is_predictive:
            not_req_cols.extend(self.other_endpoints)

        else:
            not_req_cols.extend(
                self.predictive_endpoints + [self.prog_due_to_death_col, self.exam_to_last_check_days_col])

        self.selected_df = self.selected_df.drop(columns=not_req_cols)
