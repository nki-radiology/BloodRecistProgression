import os
import pandas as pd


def get_mean_df(res, metric_cols, exp_path):
    ''' 
    Prepares a df with averaged evaluation metrics to facilitate plotting and/or displaying results
    '''
    plot_df = res[['target', 'model']].drop_duplicates()
    if "change" in exp_path:
        category = "change_in_blood_values"
    else:
        category = "blood_values"

    if "OS_from_PFS" in exp_path:
        category = "OS_from_PFS"

    if "progression" in exp_path:
        category = "current_progression"

    plot_df['exp_category'] = category

    plot_df_metrics = res[metric_cols].mean().to_frame().T.round(2)
    final_plot_df = plot_df.join(plot_df_metrics)
    return final_plot_df


def sum_blood_results(models, targets, exp_path, metric_col):
    results_dfs = []
    print(f"{metric_col}:")
    for model in models:
        print(f"{model}:")
        for target in targets:
            res_path = os.path.join(exp_path, f"{target}_{model}_results.xlsx")
            res = pd.read_excel(res_path)
            metric_cols = [metric_col + "_" + str(i) for i in [50, 2.5, 97.5]]

            mean_df = get_mean_df(res, metric_cols, exp_path)
            results_dfs.append(mean_df)
            means = {}
            for col in metric_cols:
                num_nans = res[col].isnull().sum()
                if num_nans > 0:
                    # print(f"Check res: {model}, {target} and {col}")
                    pass
                mean = round(res[col].mean(), 2)
                means[col] = mean

            min_pos = res['N_pos'].min()
            max_pos = res['N_pos'].max()

            min_neg = res['N_neg'].min()
            max_neg = res['N_neg'].max()

            min_N = res['N'].min()
            max_N = res['N'].max()

            sig_pval = res[res['pval'] < 0.05]
            max_pval = res['pval'].max()
            print(
                f"{target}: {means[metric_cols[0]]} ({means[metric_cols[1]]} - {means[metric_cols[2]]}) , pval: <0.05 ({len(sig_pval)}%) {max_pval}* N+({min_pos}, {max_pos}), N-({min_neg}, {max_neg}), N({min_N}, {max_N})")

    return results_dfs


def summarize_blood_results(exp_path="Results", models=['RF'], metric_col=['roc_auc']):
    print(os.listdir(exp_path))
    targets = os.listdir(os.path.join(exp_path, 'preds'))
    exp_path = os.path.join(exp_path, 'evaluation')

    for metric in metric_col:
        results_dfs = sum_blood_results(models, targets, exp_path, metric)
    return results_dfs
