import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from Utils import create_paths


def get_SHAP_values_df(clf, X_test, used_features, test, k, seed, model_name, target, target_shap_path):
    test=test.reset_index(drop=True)
    shap_values = add_SHAP_explanations(clf, X_test, used_features)
    shap_features = [f"shap_{feat}" for feat in used_features]
    shap_values_df = pd.DataFrame(shap_values, columns=shap_features)
    shap_values_df = pd.concat([test, shap_values_df], axis=1)
    assert len(shap_values_df)==len(test)
    shap_values_df['k'] = k
    shap_values_df['seed'] = seed
    shap_values_df['model'] = model_name
    shap_values_df['target'] = target
    shap_values_df.to_excel(
        os.path.join(target_shap_path, f"shap_values_{target}_{model_name}_{k}_{seed}.xlsx"),
        index=False)
    return shap_values_df


def add_SHAP_explanations(clf, X_test, features_names):
    explainer = shap.Explainer(clf.predict, X_test)
    shap_values_exp = explainer(X_test)
    shap_values = shap_values_exp.values
    assert len(features_names) == shap_values[0].shape[0]
    return shap_values


def create_shap_interaction_plot(avg_shap_values, features, features_names, shap_plot_path, feature_1, feature_2):
    plt.clf()
    fig = shap.dependence_plot(feature_1, avg_shap_values, features=features,
                               feature_names=features_names, interaction_index=feature_2, show=False)
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(shap_plot_path.replace('svg', 'jpg'), dpi=300, bbox_inches='tight')
    plt.close()


def create_shap_summary_plot(avg_shap_values, features, features_names, shap_plot_path):
    plt.clf()
    fig = shap.summary_plot(avg_shap_values, features=features,
                            feature_names=features_names, show=False)
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(shap_plot_path.replace('svg', 'jpg'), dpi=300, bbox_inches='tight')
    plt.close()


def average_all_shap_values(df):
    uniques = df['k'].unique()
    assert len(uniques) == len(df)
    if 'model' in df.columns:
        models = df['model'].unique()
    else:
        models = ['XGB', "RF"]

    # should be always 1 though
    assert len(models) == 1
    target = df['target'].unique()
    assert len(target) == 1

    req_columns = [x for x in list(df.columns) if "shap_" in x]
    to_remove_col = [x for x in req_columns] + ['k', 'seed'] + [x for x in list(df.columns) if "preds" in x]
    duplicated_df_cols = list(df.columns)
    for y in to_remove_col:
        duplicated_df_cols.remove(y)

    # assert it's a duplicate
    deduplicated = df[duplicated_df_cols]
    deduplicated = deduplicated.drop_duplicates()
    assert len(deduplicated) == 1
    for col in req_columns:
        deduplicated[col + '_mean'] = df[col].mean()
        deduplicated[col + '_std'] = df[col].std()
        if not 'num_reps' in deduplicated.columns:
            deduplicated['num_reps'] = len(df)

    return deduplicated


def summarize_unique_avg_shap_values(output_exp_path, models=['RF']):
    shap_path = os.path.join(output_exp_path, "SHAP")
    targets = os.listdir(shap_path)
    for target in targets:
        all_path = glob.glob(os.path.join(shap_path, target) + '/*all_100_folds.xlsx')
        assert len(all_path) <= 2
        for path in all_path:
            assert len(models) <= 2
            if f"_{models[0]}_" in path:
                model = models[0]
            else:
                model = models[1]

            df = pd.read_excel(path)

            if "change" in output_exp_path:
                task_spec_cols = ['patient_ids', 'from', 'to']
            else:
                task_spec_cols = ['patient_ids', 'start']

            avg_df = df.groupby(task_spec_cols).apply(average_all_shap_values)
            avg_df.to_excel(os.path.join(*[shap_path, target, f"averaged_unique_samples_{target}_{model}.xlsx"]),
                            index=False)


def get_shap_feature_importance(shap_values, column_names):
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(column_names, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    return feature_importance


def plot_shap_summary(output_exp_path, models=['RF'], first_routine_feature='Hb', first_tm_feature='CEA'):
    shap_path = os.path.join(output_exp_path, "SHAP")
    shap_summary_path = os.path.join(shap_path, "SHAP_summary")
    create_paths([shap_summary_path])
    targets = os.listdir(shap_path)
    targets.remove('SHAP_summary')

    feat_imp_dfs = {}
    for model_name in models:
        feat_imp_dfs[model_name] = []

    for t, target in enumerate(targets):
        all_paths = glob.glob(os.path.join(shap_path, target) + '/averaged_unique_samples*.xlsx')
        assert len(all_paths) <= 2
        for path in all_paths:
            model_name = (path.split("_")[-1]).split('.')[0]
            df = pd.read_excel(path)
            if first_routine_feature in df.columns:
                num_features = 31
                start_index = list(df.columns).index(first_routine_feature)
                if first_tm_feature in df.columns:
                    num_features += 5

            elif first_tm_feature in df.columns:
                num_features = 5
                start_index = list(df.columns).index(first_tm_feature)

            if "change" in shap_path:
                num_features += 2

            features = df.iloc[:, start_index:start_index + num_features]
            features_names = list(features.columns)
            shap_values_cols = [f"shap_{x}_mean" for x in features_names]
            avg_shap_values = df[shap_values_cols].copy().to_numpy()

            shap_feature_importance = get_shap_feature_importance(avg_shap_values, features_names)
            most_important_features = shap_feature_importance.head()['col_name'].to_list()[:5]
            print(f"{model_name} - {target}: {most_important_features}")

            feat_dict = {
                'target': target
            }

            for x, item in enumerate(most_important_features):
                feat_dict[f"{x + 1}_feature"] = item
            feat_imp_dfs[model_name].append(feat_dict)

            shap_plot_path = os.path.join(*[shap_path, target, f"shap_summary_plot_avg_{model_name}_{target}.svg"])
            create_shap_summary_plot(avg_shap_values, features, features_names, shap_plot_path)

            interaction_plots_dir = os.path.join(*[shap_path, "SHAP_summary", "shap_interaction_plot"])

            features_set = [("Lympho", "Neutr"), ("ALP", "CRP"), ("Albumin", "ALP"), ("Mono", "Lympho"),
                            ("Plt", "Lympho"), ("Neutr", "Lympho")]

            for feature_1, feature_2 in features_set:
                interaction_path = os.path.join(interaction_plots_dir, f"{feature_1}_{feature_2}")
                if not os.path.exists(interaction_path):
                    os.makedirs(interaction_path)
                shap_interaction_plot_path = os.path.join(
                    *[interaction_path, f"shap_interaction_plot_avg_{model_name}_{target}_{feature_1}_{feature_2}.svg"])
                create_shap_interaction_plot(avg_shap_values, features, features_names,
                                             shap_interaction_plot_path,
                                             feature_1, feature_2)

    for model_name in models:
        df_sum_shap = pd.DataFrame.from_dict(feat_imp_dfs[model_name])
        df_sum_shap.to_excel(os.path.join(shap_summary_path, f"{model_name}_shap_top_5_features.xlsx"))
        for col in ['1_feature', '2_feature']:
            print(df_sum_shap[col].value_counts())
