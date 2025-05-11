import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import naiveautoml

raw_df = pd.read_csv('experiments/lc_bench_results/all_datasets_lw_clean.csv')
dataset_names = raw_df['dataset_name'].unique()

dataset_nones = []
for dataset_name in dataset_names:
    print(f"Processing dataset: {dataset_name}")
    clean_df = raw_df[raw_df['dataset_name'] == dataset_name].copy()
    model_df = clean_df.drop(columns='dataset_name')

    target = 'final_val_accuracy'
    X = model_df.drop(target, axis=1)
    y = model_df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

    naml = naiveautoml.NaiveAutoML(timeout_overall=60)
    naml.fit(X_train, y_train)

    pipeline = naml.chosen_model
    # log the pipeline
    with open(f'experiments/lc_bench_results/predicted_ebe_performance/EBE_NAS_pipeline.txt', 'a') as f:
        f.write(dataset_name)
        f.write(f'{str(pipeline)}\n')
    
    if pipeline is None:
        dataset_nones.append(dataset_name)
        pass
    else:
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"Validation MSE: {mse:.4f}")
        print(f"Validation R²: {r2:.4f}")

        new_archs_df = pd.read_csv('experiments/lc_bench_results/EBE_NAS_all_results_features.csv')
        new_archs_df = new_archs_df[new_archs_df['dataset_name'] == dataset_name]
        new_archs_features = new_archs_df[X_train.columns]

        predicted_performance = pipeline.predict(new_archs_features)
        new_archs_df[target] = predicted_performance

        new_archs_df.to_csv(f'experiments/lc_bench_results/predicted_ebe_performance/EBE_NAS_predicted_{dataset_name}.csv', index=False)
    
    print(f'Models not found for {dataset_nones}')