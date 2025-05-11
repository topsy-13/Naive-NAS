import pandas as pd
import epoch_based_evolution as ebe
import load_data
import ast
import numpy as np

ebe_results = pd.read_csv(r'experiments\lc_bench_results\EBE_NAS_all_results.csv')
ebe_results['scheduler_params'] = ebe_results['scheduler_params'].apply(lambda x: ast.literal_eval(x))
ebe_results = ebe_results.fillna(np.nan).replace({np.nan: None})

# datasets = ebe_results['dataset_name'].unique().tolist()
dataset_dict = ebe_results[['dataset_name', 'dataset_id']].drop_duplicates().set_index('dataset_name')['dataset_id'].to_dict()

for i, (dataset_name, dataset_id) in enumerate(dataset_dict.items()):
    ebe_dataset = ebe_results[ebe_results['dataset_name'] == dataset_name].copy().sort_values('val_acc')
    ebe_dataset = ebe_dataset[ebe_dataset['epoch'] == max(ebe_dataset['epoch'])].head(5).reset_index(drop=True)


    print(f'Testing for Dataset {dataset_name} | {i+1}/{len(dataset_dict)}')
    X_train, y_train, X_val, y_val, X_test, y_test = load_data.get_preprocessed_data(
        dataset_id=dataset_id, scaling=True, random_seed=13, return_as='tensor')
    input_size, output_size = load_data.get_tensor_sizes(X_train, y_train)

    # Create models for each row in the dataframe
    val_accuracies = []
    for i, row in ebe_dataset.iterrows():
        print(f"Creating model for configuration {i+1}:")
        model = ebe.create_model_from_row(row, input_size, output_size)
        print(f"Model {i+1} created successfully\n")

        # Create DataLoaders
        batch_size = row['batch_size']
        train_loader = ebe.create_dataloaders(X_train, y_train, batch_size)
        val_loader = ebe.create_dataloaders(X_val, y_val, batch_size)
        test_loader = ebe.create_dataloaders(X_test, y_test, batch_size)
        # Train the model
        train_loss, train_acc = model.oe_train(train_loader, 50)
        val_loss, val_accuracy = model.evaluate(val_loader)
        val_accuracies.append(val_accuracy)
    
    ebe_dataset['50e_val_acc'] = val_accuracies

    ebe_dataset.to_csv(f'experiments/lc_bench_results/50 epochs_EBE/EBE_50-{dataset_name}.csv')