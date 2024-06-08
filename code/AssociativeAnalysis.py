import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from lifelines import CoxTimeVaryingFitter


def get_blood_df_details(blood_data):
    blood_df = blood_data.selected_df
    blood_markers = blood_data.selected_blood_markers
    all_blood_data_cols = blood_df.columns
    remaining_cols = [x for x in all_blood_data_cols if x not in blood_markers]
    df_markers = blood_df[blood_markers]
    df_rest = blood_df[remaining_cols]

    return blood_df, blood_markers, df_markers, df_rest


def normalize_blood_df(df_markers, blood_markers):
    scaler = MinMaxScaler()
    markers_scaled = scaler.fit_transform(df_markers)
    df_markers_scaled = pd.DataFrame(markers_scaled, columns=blood_markers)
    return markers_scaled, df_markers_scaled


def run_cox_time_varying_analysis(df, output_exp_path, event_col, stop_col, start_col='start', id_col='patient_ids'):
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(df, id_col=id_col,
            event_col=event_col,
            start_col=start_col,
            stop_col=stop_col,
            show_progress=True)
    ctv.print_summary()
    ctv.plot()

    pred = ctv.predict_partial_hazard(df)
    plt.savefig(os.path.join(output_exp_path, f'{event_col}_associate.jpg'))
    return ctv.summary
