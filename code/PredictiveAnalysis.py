import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import Utils
from StatsUtils import diagnostic_performance_analysis
from Utils import create_paths, find_intersection, pickle_data, read_pickled_data
from ShapUtils import get_SHAP_values_df, summarize_unique_avg_shap_values, plot_shap_summary


def get_clf(model_name, gkf, hyp_param_n_iter):
    if "RF" in model_name:
        clf = get_RF_clf(gkf, hyp_param_n_iter)
    elif "XGB" in model_name:
        clf = get_XGB_clf(gkf, hyp_param_n_iter=1)
    elif "LR" in model_name:
        clf = get_LR_clf(gkf, hyp_param_n_iter=1)
    elif "SVM" in model_name:
        clf = get_SVM_clf(gkf, hyp_param_n_iter=1)

    return clf


def get_XGB_clf(gkf, hyp_param_n_iter):
    clf = XGBClassifier(random_state=42, use_label_encoder=False)

    params_random = {'gamma': [0, 0.5, 1, 1.5, 2, 5],
                     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                     'colsample_bytree': [0.6, 0.8, 1.0],
                     'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     'n_estimators': [1, 50, 100, 150, 200],
                     'learning_rate': [0.0001, 0.001, 0.01, 0.1]}
    params_random = {}
    clf = RandomizedSearchCV(clf,
                             param_distributions=params_random,
                             n_iter=hyp_param_n_iter,
                             random_state=42,
                             cv=gkf)

    return clf


def get_LR_clf(gkf, hyp_param_n_iter):
    clf = LogisticRegression(random_state=0)
    random_grid = {  # without hyperparameter optimization
    }

    clf = RandomizedSearchCV(clf,
                             param_distributions=random_grid,
                             n_iter=hyp_param_n_iter,
                             cv=gkf,
                             random_state=0)
    return clf


def get_SVM_clf(gkf, hyp_param_n_iter):
    clf = SVC(kernel='rbf', probability=True, random_state=0)
    random_grid = {  # without hyperparameter optimization
    }

    clf = RandomizedSearchCV(clf,
                             param_distributions=random_grid,
                             n_iter=hyp_param_n_iter,
                             cv=gkf,
                             random_state=0)
    return clf


def get_RF_clf(gkf, hyp_param_n_iter):
    clf = RandomForestClassifier(random_state=0)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=10)] + [300]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # class_weights = ["balanced"]  # , "balanced_subsample"]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   }

    clf = RandomizedSearchCV(clf,
                             param_distributions=random_grid,
                             n_iter=hyp_param_n_iter,
                             cv=gkf,
                             # scoring='roc_auc',
                             random_state=0)
    return clf


def run_ML_analysis(blood_data_instance, seeds, hyp_param_n_iter=50, do_random_optim=1, models=["RF"],
                    enable_shap=False):
    blood_data = blood_data_instance.selected_df
    targets = blood_data_instance.targets
    output_exp_path = blood_data_instance.output_exp_path
    os_from_pfs = blood_data_instance.os_from_pfs

    pfs_endpoints = blood_data_instance.pfs_endpoints
    os_endpoints = blood_data_instance.os_endpoints
    all_targets = pfs_endpoints + os_endpoints + [blood_data_instance.current_progression]

    results_saving_path = os.path.join(output_exp_path, 'evaluation')
    feat_imp_saving_path = os.path.join(output_exp_path, 'feature_importances')
    figures_saving_path = os.path.join(output_exp_path, 'plots')
    models_saving_path = os.path.join(output_exp_path, 'models')
    data_saving_path = os.path.join(output_exp_path, 'data')
    preds_path = os.path.join(output_exp_path, 'preds')
    shap_saving_path = os.path.join(output_exp_path, "SHAP")
    create_paths(
        [results_saving_path, feat_imp_saving_path, figures_saving_path, models_saving_path, data_saving_path])

    for model_name in models:
        if "RF" in model_name:
            clf_feat_importances_df = {}

            for target in targets:
                clf_feat_importances_df[target] = pd.DataFrame(index=range(len(seeds)))

            feat_importances_dfs = {
                model_name: clf_feat_importances_df
            }

    for target in targets:
        all_preds_dfs, all_os_preds_dfs = [], []
        all_shap_values_dfs, all_os_shap_values_dfs = [], []
        target_preds_path = os.path.join(preds_path, target)
        target_shap_path = os.path.join(shap_saving_path, target)
        create_paths([target_preds_path, target_shap_path])
        if os_from_pfs:
            os_target = target.replace('pfs', 'os')
            os_output_exp_path = f"{output_exp_path}_OS_from_PFS"
            os_target_preds_path = os.path.join(*[os_output_exp_path, 'preds', os_target])
            os_target_shap_path = os.path.join(*[os_output_exp_path, 'SHAP', os_target])
            os_eval_path = os.path.join(os_output_exp_path, 'evaluation')
            create_paths([os_eval_path, os_target_preds_path, os_target_shap_path])

        AUCs, all_results, all_os_results = {}, {}, {}
        for model_name in models:
            AUCs[model_name] = []
            all_results[model_name] = []
            all_os_results[model_name] = []

        print('\n', target)

        # remove tests that are lost to FU within the prediction timepoint
        if 'pfs' in target:
            target_days = blood_data_instance.pred_days_from_SoT[target.replace('pfs_', '')]
            included_blood_data = blood_data[blood_data['exam_to_last_check_days'] >= target_days]
        else:
            included_blood_data = blood_data

        for k, seed in enumerate(seeds):
            if seed % 25 == 0:
                print(f"fold: {seed}")

            b_data = included_blood_data

            if any(x == target for x in pfs_endpoints):
                b_data = b_data.drop_duplicates(subset=["patient_ids", "stop_pfs"])

            if ((blood_data_instance.include_only_alive_patients) and (('pfs' in target) or ('progression' in target))):
                death_col = target.replace('pfs', 'os')
                b_data = b_data[b_data[death_col] == 0]  

            splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=seed)
            split = splitter.split(b_data, groups=b_data['patient_ids'])
            train_inds, test_inds = next(split)

            train = b_data.iloc[train_inds]
            test = b_data.iloc[test_inds]

            train_ids = train['patient_ids'].unique()
            test_ids = test['patient_ids'].unique()
            assert len(find_intersection(train_ids, test_ids)) == 0

            train.to_excel(os.path.join(data_saving_path, f"train_data_{target}_{k}_{seed}.xlsx"), index=False)
            test.to_excel(os.path.join(data_saving_path, f"test_data_{target}_{k}_{seed}.xlsx"), index=False)

            cols_to_drop = ['patient_ids', "stop_pfs", 'stop', 'start', 'prog_due_to_death',
                            'exam_to_last_check_days'] + all_targets
            X_train = train.drop(columns=cols_to_drop)
            y_train = np.array(train[target].astype(int))
            # print(np.unique(y_train))
            X_test = test.drop(columns=cols_to_drop)
            y_test = np.array(test[target].astype(int))
            # print(np.unique(y_test))
            if os_from_pfs:
                y_test_os = np.array(test[os_target].astype(int))

            used_features = list(X_test.columns)

            gkf = list(GroupKFold(n_splits=5).split(X_train, y_train, train['patient_ids']))
            for model_name in models:
                clf = get_clf(model_name, gkf, hyp_param_n_iter)
                clf.fit(X_train, y_train)
                trained_models = {model_name: clf}

            # evaluate trained models
            for model_name, clf in trained_models.items():
                y_pred_probs = clf.predict_proba(X_test)[:, 1]

                if os_from_pfs:
                    os_metrics, os_preds_df, os_shap_values_df = predict_os_from_pfs_models(os_target, clf, X_test,
                                                                                            y_test_os, test,
                                                                                            enable_shap,
                                                                                            os_target_preds_path,
                                                                                            os_target_shap_path, k,
                                                                                            seed, model_name)
                    all_os_shap_values_dfs.append(os_shap_values_df)
                    all_os_preds_dfs.append(os_preds_df)
                    all_os_results[model_name].append(os_metrics)

                if enable_shap:
                    shap_values_df = get_SHAP_values_df(clf, X_test, used_features, test, k, seed, model_name, target,
                                                        target_shap_path)
                    all_shap_values_dfs.append(shap_values_df)

                preds_df = test.copy()
                preds_df[f'preds_{target}_{model_name}'] = y_pred_probs

                metrics = diagnostic_performance_analysis(np.array(y_test), np.array(y_pred_probs), model_name, target,
                                                          k + 1)
                AUCs[model_name].append(metrics['roc_auc_50'])
                all_results[model_name].append(metrics)
                pickle_data(clf, os.path.join(models_saving_path, f"{model_name}_{target}_{k}_{seed}.pickle"))

                if "RF" in model_name:
                    imp = clf.best_estimator_.feature_importances_ if do_random_optim else clf.feature_importances_
                    for g in range(len(X_train.columns)):
                        for endpoint in targets:
                            if target == endpoint:
                                feat_importances_dfs[model_name][endpoint].loc[k, X_train.columns[g]] = imp[g]

                    save_feature_importances(feat_importances_dfs, feat_imp_saving_path)

            # prepare to save the preds of both models added to test df for the specific target
            preds_df['k'] = k
            preds_df['seed'] = seed
            all_preds_dfs.append(preds_df)
            del X_train, y_train, train, X_test, test, y_test, preds_df, clf

        # save predictions probabilities for further analysis
        all_preds_dfs = pd.concat(all_preds_dfs)
        all_preds_dfs.to_excel(os.path.join(target_preds_path, f"test_preds_{target}_all_100_folds.xlsx"),
                               index=False)
        if os_from_pfs:
            all_os_preds_dfs = pd.concat(all_os_preds_dfs)
            all_os_preds_dfs.to_excel(os.path.join(os_target_preds_path, f"test_preds_{os_target}_all_100_folds.xlsx"),
                                      index=False)

            os_results_df = pd.DataFrame.from_dict(all_os_results[model_name])
            os_cols = os_results_df.columns.tolist()
            os_ordered_cols = os_cols[-3:] + os_cols[:-3]
            os_results_df = os_results_df[os_ordered_cols]
            os_results_df.to_excel(os.path.join(os_eval_path, f"{os_target}_{model_name}_results.xlsx"), index=False)

        # save shap values of all exps
        if enable_shap:
            all_shap_values_dfs = pd.concat(all_shap_values_dfs)
            all_shap_values_dfs.to_excel(
                os.path.join(target_shap_path, f"shap_values_{target}_{model_name}_all_100_folds.xlsx"), index=False)
            if os_from_pfs:
                all_os_shap_values_dfs = pd.concat(all_os_shap_values_dfs)
                all_os_shap_values_dfs.to_excel(
                    os.path.join(os_target_shap_path, f"shap_values_{os_target}_{model_name}_all_100_folds.xlsx"),
                    index=False)

        for model_name in trained_models.keys():
            print(f"{model_name} - {target}: Mean: {np.nanmean(AUCs[model_name])}")
            results_df = pd.DataFrame.from_dict(all_results[model_name])
            cols = results_df.columns.tolist()
            ordered_cols = cols[-3:] + cols[:-3]
            results_df = results_df[ordered_cols]
            results_df.to_excel(os.path.join(results_saving_path, f"{target}_{model_name}_results.xlsx"), index=False)

    if enable_shap:
        summarize_unique_avg_shap_values(output_exp_path)
        plot_shap_summary(output_exp_path)

        if os_from_pfs:
            summarize_unique_avg_shap_values(os_output_exp_path)
            plot_shap_summary(os_output_exp_path)


def predict_pfs(trained_models_folder, blood_data_instance, seeds, models=["RF"], enable_shap=False):
    blood_data = blood_data_instance.selected_df
    targets = blood_data_instance.targets
    output_exp_path = blood_data_instance.output_exp_path
    os_from_pfs = blood_data_instance.os_from_pfs

    pfs_endpoints = blood_data_instance.pfs_endpoints
    os_endpoints = blood_data_instance.os_endpoints
    all_targets = pfs_endpoints + os_endpoints + [blood_data_instance.current_progression]

    results_saving_path = os.path.join(output_exp_path, 'evaluation')
    feat_imp_saving_path = os.path.join(output_exp_path, 'feature_importances')
    figures_saving_path = os.path.join(output_exp_path, 'plots')
    data_saving_path = os.path.join(output_exp_path, 'data')
    preds_path = os.path.join(output_exp_path, 'preds')
    shap_saving_path = os.path.join(output_exp_path, "SHAP")
    create_paths(
        [results_saving_path, feat_imp_saving_path, figures_saving_path, data_saving_path])

    for model_name in models:
        if "RF" in model_name:
            clf_feat_importances_df = {}

            for target in targets:
                clf_feat_importances_df[target] = pd.DataFrame(index=range(len(seeds)))

            feat_importances_dfs = {
                model_name: clf_feat_importances_df
            }

    for target in targets:
        all_preds_dfs, all_os_preds_dfs = [], []
        all_shap_values_dfs, all_os_shap_values_dfs = [], []
        target_preds_path = os.path.join(preds_path, target)
        target_shap_path = os.path.join(shap_saving_path, target)
        create_paths([target_preds_path, target_shap_path])
        if os_from_pfs:
            os_target = target.replace('pfs', 'os')
            os_output_exp_path = f"{output_exp_path}_OS_from_PFS"
            os_target_preds_path = os.path.join(*[os_output_exp_path, 'preds', os_target])
            os_target_shap_path = os.path.join(*[os_output_exp_path, 'SHAP', os_target])
            os_eval_path = os.path.join(os_output_exp_path, 'evaluation')
            create_paths([os_eval_path, os_target_preds_path, os_target_shap_path])

        AUCs, all_results, all_os_results = {}, {}, {}
        for model_name in models:
            AUCs[model_name] = []
            all_results[model_name] = []
            all_os_results[model_name] = []

        print('\n', target)

        for k, seed in enumerate(seeds):
            if seed % 25 == 0:
                print(f"fold: {seed}")

            b_data = blood_data

            if any(x == target for x in pfs_endpoints):
                b_data = b_data.drop_duplicates(subset=["patient_ids", "stop_pfs"])

            splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=seed)
            split = splitter.split(b_data, groups=b_data['patient_ids'])
            train_inds, test_inds = next(split)

            train = b_data.iloc[train_inds]
            test = b_data.iloc[test_inds]

            # if ((blood_data_instance.exclude_prog_due_to_death) and (('pfs' in target) or ('progression' in target))):
            #     b_data = b_data[~b_data['prog_due_to_death']]

            if ((blood_data_instance.include_only_alive_patients) and (('pfs' in target) or ('progression' in target))):
                death_col = target.replace('pfs', 'os')
                train = train[train[death_col] == 0]
                test = test[test[death_col] == 0]
                # print(f"Unique patients in the study after reducing patients experiencing death as progression: train: {train['patient_ids'].nunique}, test: {test['patient_ids'].nunique}")

            train_ids = train['patient_ids'].unique()
            test_ids = test['patient_ids'].unique()
            assert len(find_intersection(train_ids, test_ids)) == 0

            train.to_excel(os.path.join(data_saving_path, f"train_data_{target}_{k}_{seed}.xlsx"), index=False)
            test.to_excel(os.path.join(data_saving_path, f"test_data_{target}_{k}_{seed}.xlsx"), index=False)

            cols_to_drop = ['patient_ids', "stop_pfs", 'stop', 'start'] + all_targets
            X_train = train.drop(columns=cols_to_drop)
            y_train = np.array(train[target].astype(int))
            X_test = test.drop(columns=cols_to_drop)
            y_test = np.array(test[target].astype(int))

            if os_from_pfs:
                y_test_os = np.array(test[os_target].astype(int))

            used_features = list(X_test.columns)

            gkf = list(GroupKFold(n_splits=5).split(X_train, y_train, train['patient_ids']))

            for model_name in models:
                model_path = os.path.join(trained_models_folder, f'{model_name}_{target}_{k}_{seed}.pickle')
                clf = read_pickled_data(model_path)

                saved_train = pd.read_excel(os.path.join(trained_models_folder.replace('models', 'data'),
                                                         f'train_data_{target}_{k}_{seed}.xlsx'))
                saved_test = pd.read_excel(os.path.join(trained_models_folder.replace('models', 'data'),
                                                        f'test_data_{target}_{k}_{seed}.xlsx'))

                if ((blood_data_instance.include_only_alive_patients) and ('pfs' in target)):
                    death_col = target.replace('pfs', 'os')
                    saved_train = saved_train[saved_train[death_col] == 0]
                    saved_test = saved_test[saved_test[death_col] == 0]

                    assert saved_train.shape == train.shape
                    assert saved_test.shape == test.shape

                y_pred_probs = clf.predict_proba(X_test)[:, 1]

                if os_from_pfs:
                    os_metrics, os_preds_df, os_shap_values_df = predict_os_from_pfs_models(os_target, clf, X_test,
                                                                                            y_test_os, test,
                                                                                            enable_shap,
                                                                                            os_target_preds_path,
                                                                                            os_target_shap_path, k,
                                                                                            seed, model_name)
                    all_os_shap_values_dfs.append(os_shap_values_df)
                    all_os_preds_dfs.append(os_preds_df)
                    all_os_results[model_name].append(os_metrics)

                if enable_shap:
                    shap_values_df = get_SHAP_values_df(clf, X_test, used_features, test, k, seed, model_name, target,
                                                        target_shap_path)
                    all_shap_values_dfs.append(shap_values_df)

                preds_df = test.copy()
                preds_df[f'preds_{target}_{model_name}'] = y_pred_probs

                metrics = diagnostic_performance_analysis(np.array(y_test), np.array(y_pred_probs), model_name, target,
                                                          k + 1)
                AUCs[model_name].append(metrics['roc_auc_50'])
                all_results[model_name].append(metrics)

            # prepare to save the preds of both models added to test df for the specific target
            preds_df['k'] = k
            preds_df['seed'] = seed
            all_preds_dfs.append(preds_df)
            del X_train, y_train, train, X_test, test, y_test, preds_df, clf

        # save predictions probabilities for further analysis
        all_preds_dfs = pd.concat(all_preds_dfs)
        all_preds_dfs.to_excel(os.path.join(target_preds_path, f"test_preds_{target}_all_100_folds.xlsx"),
                               index=False)
        if os_from_pfs:
            all_os_preds_dfs = pd.concat(all_os_preds_dfs)
            all_os_preds_dfs.to_excel(os.path.join(os_target_preds_path, f"test_preds_{os_target}_all_100_folds.xlsx"),
                                      index=False)

            os_results_df = pd.DataFrame.from_dict(all_os_results[model_name])
            os_cols = os_results_df.columns.tolist()
            os_ordered_cols = os_cols[-3:] + os_cols[:-3]
            os_results_df = os_results_df[os_ordered_cols]
            os_results_df.to_excel(os.path.join(os_eval_path, f"{os_target}_{model_name}_results.xlsx"), index=False)

        # save shap values of all exps
        if enable_shap:
            all_shap_values_dfs = pd.concat(all_shap_values_dfs)
            all_shap_values_dfs.to_excel(
                os.path.join(target_shap_path, f"shap_values_{target}_{model_name}_all_100_folds.xlsx"), index=False)
            if os_from_pfs:
                all_os_shap_values_dfs = pd.concat(all_os_shap_values_dfs)
                all_os_shap_values_dfs.to_excel(
                    os.path.join(os_target_shap_path, f"shap_values_{os_target}_{model_name}_all_100_folds.xlsx"),
                    index=False)

        for model_name in models:
            print(f"{model_name} - {target}: Mean: {np.nanmean(AUCs[model_name])}")
            results_df = pd.DataFrame.from_dict(all_results[model_name])
            cols = results_df.columns.tolist()
            ordered_cols = cols[-3:] + cols[:-3]
            results_df = results_df[ordered_cols]
            results_df.to_excel(os.path.join(results_saving_path, f"{target}_{model_name}_results.xlsx"), index=False)

    if enable_shap:
        summarize_unique_avg_shap_values(output_exp_path)
        plot_shap_summary(output_exp_path)

        if os_from_pfs:
            summarize_unique_avg_shap_values(os_output_exp_path)
            plot_shap_summary(os_output_exp_path)


def get_saved_preds(prev_exp_path, results_saving_path, interval_start, interval_end, target, model_name="RF"):
    Utils.create_paths([results_saving_path])
    AUCs = {}
    all_results = {}
    AUCs[model_name] = []
    all_results[model_name] = []

    preds_path = f"{prev_exp_path},preds,{target},test_preds_{target}_all_100_folds.xlsx".split(",")
    df = pd.read_excel(os.path.join(*preds_path))
    df = df[(df.start > interval_start) & (df.start <= interval_end)]
    df = df.reset_index(drop=True)
    for fold in range(100):
        df = df[df['k'] == fold]
        df = df.reset_index(drop=True)
        y_pred_probs = df[f'preds_{target}_{model_name}'].to_numpy()
        y_test = df[target].to_numpy()
        metrics = diagnostic_performance_analysis(y_test, y_pred_probs, model_name, target, "")
        AUCs[model_name].append(metrics['roc_auc_50'])
        all_results[model_name].append(metrics)
    print(f"{model_name} - {target}: Mean: {np.nanmean(AUCs[model_name])}")
    results_df = pd.DataFrame.from_dict(all_results[model_name])
    cols = results_df.columns.tolist()
    ordered_cols = cols[-3:] + cols[:-3]
    results_df = results_df[ordered_cols]
    results_df.to_excel(os.path.join(results_saving_path,
                                     f"{target}_{model_name}_results_interval_{interval_start}_{interval_end}.xlsx"),
                        index=False)


def save_feature_importances(feat_importances_dfs, feat_imp_saving_path):
    for model_name, model_feat_imp_df in feat_importances_dfs.items():
        for target_name, target_feat_imp_df in model_feat_imp_df.items():
            target_feat_imp_df.to_excel(
                os.path.join(feat_imp_saving_path, f"features_importances_{target_name}_{model_name}.xlsx"),
                index=False)

        # plot_feature_importances(feat_importances_dfs, targets, model_name, figures_saving_path)


def predict_os_from_pfs_models(os_target, clf, X_test, y_test_os, os_test_df, enable_shap, os_target_preds_path,
                               os_target_shap_path, k, seed, model_name):
    used_features = list(X_test.columns)
    y_pred_probs = clf.predict_proba(X_test)[:, 1]
    os_test_df = os_test_df.reset_index(drop=True)
    os_preds_df = os_test_df.copy()
    os_preds_df.loc[:, f'preds_{os_target}_{model_name}'] = y_pred_probs
    os_preds_df.loc[:, 'k'] = k
    os_preds_df.loc[:, 'seed'] = seed
    metrics = diagnostic_performance_analysis(np.array(y_test_os), np.array(y_pred_probs), model_name, os_target,
                                              k + 1)

    if enable_shap:
        shap_values_df = get_SHAP_values_df(clf, X_test, used_features, os_test_df, k, seed, model_name, os_target,
                                            os_target_shap_path)
    else:
        shap_values_df = None
    return metrics, os_preds_df, shap_values_df
