# the included data types in the study
# 'blood_data_types': ['ROUTINE', 'TUMOR_MARKERS', 'ROUTINE_AND_TUMOR_MARKERS'],

# contains the names of columns of the data provided in an excel file
dataset_params = {
    'blood_markers_files_names_dict': {
        'ROUTINE': 'dataset_routine_blood.xlsx',
        'TUMOR_MARKERS': 'dataset_tumor_markers.xlsx',
        'ROUTINE_AND_TUMOR_MARKERS': 'dataset_routine_and_tumor_markers.xlsx'
    },

    'first_feature_col_names_dict': {
        'ROUTINE': 'Hemoglobine',
        'TUMOR_MARKERS': 'CEA',
        'ROUTINE_AND_TUMOR_MARKERS': 'Hemoglobine'
    },
    # dataset columns
    'id_col': 'anonym_id',
    'exam_date_col': 'exam_date',
    'death_date_col': 'death_date',
    'prog_due_to_death_col': 'prog_due_to_death',
    'treatment_date_col': 'treatment_date',
    'last_check_date_col': 'last_check_date',
    'pfs_event_col': 'pfs',
    'progression_date_col': 'progression_date',
    'pfs_days_from_SoT_col': 'pfs_days_from_SoT',
    'diff_death_from_prog_col': 'diff_death_from_prog',
    'os_days_from_exam_date_col': 'os_days_from_exam_date',
    'lab_test_from_SoT_col': 'lab_test_from_SoT',

    'age_col': 'clin.age',
    'sex_col': 'clin.sex',
    # cols to calculate
    'updated_id_col': 'patient_ids',
    'os_event_col': 'os',
    'exam_to_last_check_days_col': 'exam_to_last_check_days',

    # label columns
    'pfs_endpoints_cols': ['pfs_1mon', 'pfs_3mon', 'pfs_6mon', 'pfs_9mon', 'pfs_1year'],
    'os_endpoints_cols': ['os_1mon', 'os_3mon', 'os_6mon', 'os_9mon', 'os_1year'],
    'current_progression_col': 'is_progression',
    'association_endpoints_cols': ['os', 'pfs'],

    'pred_days_from_SoT': {
        '1mon': 31,
        '3mon': 92,
        '6mon': 183,
        '9mon': 274,
        '1year': 365
    },

    'features_names_abv_mapping': {
        'Hemoglobine': 'Hb',
        'Hematocriet': 'Ht',
        'MCV': 'MCV',
        'Erytrocyten': 'RBC',
        'Trombocyten': 'Plt',
        'Leukocyten': 'WBC',
        'Dc_Lymfo': 'Lympho',
        'Dc_Mono': 'Mono',
        'Dc_Eos': 'Eos',
        'Dc_Baso': 'Baso',
        'Dc_Neutr': 'Neutr',
        'Dc_NeutrGran': 'NeutrGran',
        'Dc_ImmGran': 'ImmGran',
        'CRP': 'CRP',
        'BiliTot': 'TBIL',
        'AF': 'ALP',
        'ASAT': 'AST',
        'ALAT': 'ALT',
        'Kreatinine': 'Cr',
        'GFR_c': 'GFR',
        'Natrium': 'Sodium',
        'Kalium': 'Potassium',
        'Chloride': 'Chloride',
        'Bicarbonaat': 'Bicarbonate',
        'Fosfaat': 'Phosphate',
        'Calcium': 'Calcium',
        'Magnesium': 'Magnesium',
        'Ureum': 'Urea',
        'Glucose': 'Glucose',
        'TE': 'Total Protein',
        'Albumine': 'Albumin',
        'from': 'start of interval',
        'to': 'end of interval'

    }
}
