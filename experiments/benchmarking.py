from api import Benchmark
import numpy as np
import pandas as pd

bench_dir = "../six_datasets_lw/six_datasets_lw.json"
dataset_name = 'Fashion-MNIST'

def get_all_accuracies(bench_dir, dataset_name):
    bench = Benchmark(bench_dir, cache=False)

    # Extract number of configs
    n_configs = bench.get_number_of_configs(dataset_name=dataset_name)

    val_balanced_acc = []
    for i in range(n_configs):
        val_balanced_acc.append(bench.query(dataset_name=dataset_name, tag='final_val_balanced_accuracy', config_id=i))
    return val_balanced_acc

def random_sample(bench_dir, dataset_name, n_samples):
    bench = Benchmark(bench_dir, cache=False)

    # Extract number of configs
    n_configs = bench.get_number_of_configs(dataset_name=dataset_name)

    # Randomly sample n_samples from the configs
    sampled_indices = np.random.choice(n_configs, size=n_samples, replace=False)
    sampled_accuracies = [bench.query(dataset_name=dataset_name, tag='final_val_balanced_accuracy', config_id=i) for i in sampled_indices]
    
    return sampled_accuracies

def build_dataframe(bench, dataset_name):
    

    # Extract number of configs
    n_configs = bench.get_number_of_configs(dataset_name=dataset_name)
    queriable_tags = bench.get_queriable_tags(dataset_name=dataset_name)

    # Create a dictionary to hold the data
    data = []
    for n_config in range(n_configs):
        tag_data = {}
        for tag in queriable_tags:
            tag_data[tag] = bench.query(dataset_name, tag, config_id=n_config)
        data.append(tag_data)
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    return df