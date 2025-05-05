import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import math

import load_data
import gc

import pandas as pd
import copy

def set_seed(seed=13):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (if available)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables auto-optimization for conv layers (useful for exact reproducibility)
    return


# Definir una arquitectura de red flexible 
import torch
import torch.nn as nn
import torch.optim as optim

class DynamicNN(nn.Module):  # MLP
    def __init__(self, input_size, output_size, 
                 hidden_layers, 
                 activation_fn, dropout_rate,
                 lr, optimizer_type, 
                 weight_decay=0, momentum=None,
                 use_skip_connections=False,
                 initializer='xavier_uniform', lr_scheduler='none',
                 scheduler_params={},
                 device=None):
        super(DynamicNN, self).__init__()

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_skip_connections = use_skip_connections
        
        layers = []
        prev_size = input_size

        for size in hidden_layers:
            layer = nn.Linear(prev_size, size)

            # Apply initializer
            if initializer == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight)
            elif initializer == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight)
            elif initializer == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight)
            elif initializer == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight)

            layers.append(layer)
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        # Configure optimizer
        optimizer_kwargs = {'lr': lr}
        if weight_decay > 0:
            optimizer_kwargs['weight_decay'] = weight_decay
        if momentum is not None and optimizer_type == optim.SGD:
            optimizer_kwargs['momentum'] = momentum

        self.optimizer = optimizer_type(self.parameters(), **optimizer_kwargs)

        # Scheduler
        if lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        elif lr_scheduler == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_params.get('gamma', 0.9)
            )
        elif lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get('T_max', 10)
            )
        else:
            self.scheduler = None

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        if not self.use_skip_connections:
            return self.network(x)

        result = x
        idx = 0

        for module in self.network:
            if isinstance(module, nn.Linear) and idx > 0:
                output = module(result)
                if result.shape == output.shape:
                    result = output + result
                else:
                    result = output
            else:
                result = module(result)
            idx += 1

        return result

        
    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()


    def oe_train(self, train_loader, num_epochs=1):
        self.train()
        
        for epoch in range(num_epochs):
            total = 0
            correct = 0
            running_loss = 0.0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                # Automatically flatten if it's an image (i.e., has more than 2 dimensions)
                if features.dim() > 2:  
                    features = features.view(features.size(0), -1)  # Flatten images

                self.optimizer.zero_grad()
                outputs = self(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                running_loss += loss.item() * features.size(0)

                # Compute accuracy
                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Compute final loss and accuracy for the epoch
            train_loss = running_loss / total
            train_acc = correct / total
            
        return train_loss, train_acc
    
    def es_train(self, train_loader, val_loader, es_patience=50, max_epochs=300, verbose=False):
        best_val_acc = -float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        best_train_loss = None
        best_train_acc = None
        best_val_loss = None

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = self.oe_train(train_loader)

            self.eval()
            running_loss_val = 0.0
            total_val = 0
            correct_val = 0

            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)

                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)

                    outputs = self(features)
                    loss = self.criterion(outputs, labels)

                    running_loss_val += loss.item() * features.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss = running_loss_val / total_val
            val_acc = correct_val / total_val

            self.train()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_model_state = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
                print('New best acc found:', best_val_acc)
            else:
                epochs_without_improvement += 1

            if verbose:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            if epochs_without_improvement >= es_patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch} epochs.")
                break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return best_train_loss, best_train_acc, best_val_loss, best_val_acc

    def evaluate(self, val_loader):
            self.eval()
            
            total = 0
            correct = 0
            running_loss = 0.0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)

                    # Automatically flatten if it's an image (i.e., has more than 2 dimensions)
                    if features.dim() > 2:  
                        features = features.view(features.size(0), -1)

                    outputs = self(features)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item() * features.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = running_loss / total
            val_accuracy = correct / total
            
            return val_loss, val_accuracy


# region Search Space
class SearchSpace():
    
    def __init__(self, input_size, output_size, 
                 min_layers=2, max_layers=7, 
                 min_neurons=13, max_neurons=512,
                 activation_fns=[nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.GELU],
                 dropout_rates=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                 min_learning_rate=0.0001, max_learning_rate=0.1,
                 min_batch_size=32, max_batch_size=1024,
                 weight_decays=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                 momentum_values=[0.8, 0.9, 0.95, 0.99],
                 layer_norm_options=[True, False],
                 skip_connection_options=[True, False],
                 initializers=['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'],
                 lr_schedulers=['step', 'exponential', 'cosine', 'none']):
        
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [min_layers, max_layers]
        self.neurons = [min_neurons, max_neurons]
        self.activation_fns = activation_fns
        
        # Store log bounds for learning rate
        self.log_min_lr = math.log10(min_learning_rate)
        self.log_max_lr = math.log10(max_learning_rate)
        
        self.dropout_rates = dropout_rates
        self.optimizers = [optim.Adam, optim.SGD, optim.RMSprop, optim.AdamW]
        self.weight_decays = weight_decays
        self.momentum_values = momentum_values
        self.layer_norm_options = layer_norm_options
        self.skip_connection_options = skip_connection_options
        self.initializers = initializers
        self.lr_schedulers = lr_schedulers

        # Build batch sizes considering powers of 2
        power = 1
        self.batch_sizes = []
        while power <= max_batch_size:
            if power >= min_batch_size:
                self.batch_sizes.append(power)
            power *= 2


    def sample_architecture(self):
        hidden_layers = random.choices(range(self.neurons[0], self.neurons[1]), 
                                       k=random.randint(self.layers[0], self.layers[1]))
        activation_fn = random.choice(self.activation_fns)
        dropout_rate = random.choice(self.dropout_rates)
        optimizer_type = random.choice(self.optimizers)
        
        # Sample learning rate on a logarithmic scale
        log_lr = random.uniform(self.log_min_lr, self.log_max_lr)
        learning_rate = 10 ** log_lr
        
        # Sample weight decay on logarithmic scale if it's not zero
        weight_decay = random.choice(self.weight_decays)
        momentum = random.choice(self.momentum_values) if optimizer_type in [optim.SGD] else None
        
        # Sample other parameters
        batch_size = random.choice(self.batch_sizes)
        use_layer_norm = random.choice(self.layer_norm_options)
        use_skip_connections = random.choice(self.skip_connection_options)
        initializer = random.choice(self.initializers)
        lr_scheduler = random.choice(self.lr_schedulers)
        
        # Sample hyperparameters specific to schedulers
        scheduler_params = {}
        if lr_scheduler == 'step':
            scheduler_params['step_size'] = random.choice([5, 10, 20, 30])
            scheduler_params['gamma'] = random.choice([0.1, 0.5, 0.9])
        elif lr_scheduler == 'exponential':
            scheduler_params['gamma'] = random.choice([0.9, 0.95, 0.99])
        elif lr_scheduler == 'cosine':
            scheduler_params['T_max'] = random.choice([10, 50, 100])
        
        return {
            'hidden_layers': hidden_layers,
            'activation_fn': activation_fn,
            'dropout_rate': dropout_rate,
            'optimizer_type': optimizer_type,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'batch_size': batch_size,
            'use_skip_connections': use_skip_connections,
            'initializer': initializer,
            'lr_scheduler': lr_scheduler,
            'scheduler_params': scheduler_params
        }

    def create_model(self, architecture):
        hidden_layers = architecture["hidden_layers"]
        activation_fn = architecture["activation_fn"]
        dropout_rate = architecture["dropout_rate"]
        optimizer_type = architecture["optimizer_type"]
        learning_rate = architecture["learning_rate"]
        self.batch_size = architecture["batch_size"]  # extract the batch size for dataloader
        
        # Extract new parameters with defaults if not present (for backward compatibility)
        weight_decay = architecture.get("weight_decay", 0)
        momentum = architecture.get("momentum", None)
        use_skip_connections = architecture.get("use_skip_connections", False)
        initializer = architecture.get("initializer", "xavier_uniform")
        lr_scheduler = architecture.get("lr_scheduler", "none")
        scheduler_params = architecture.get("scheduler_params", {})
        
        # Create model with all parameters
        model = DynamicNN(
            self.input_size, self.output_size, 
            hidden_layers, activation_fn, 
            dropout_rate, learning_rate, optimizer_type,
            weight_decay=weight_decay,
            momentum=momentum,
            use_skip_connections=use_skip_connections,
            initializer=initializer,
            lr_scheduler=lr_scheduler,
            scheduler_params=scheduler_params,
            device=self.device
        ).to(self.device)
        
        return model
    
# endregion

# region Generations
class Generation():
    def __init__(self, search_space, n_individuals):
        self.search_space = search_space
        self.n_individuals = n_individuals
        self.generation = self.build_generation() 

    def build_generation(self):
        generation = {}
        for i in range(self.n_individuals):
            architecture = self.search_space.sample_architecture()
            model = self.search_space.create_model(architecture)
            generation[i] = {
                "model": model,
                "architecture": architecture,
                "batch_size": architecture['batch_size']
            }
        return generation
    
    def train_generation(self, X_train, y_train, num_epochs=1):
        for i in range(self.n_individuals):
            model = self.generation[i]["model"]
            batch_size = self.generation[i]["batch_size"]
            # Create a DataLoader with the architecture-specific batch size
            train_loader = create_dataloaders(X=X_train, y=y_train, batch_size=batch_size)

            train_loss, train_acc = model.oe_train(train_loader, num_epochs=num_epochs)
            self.generation[i]["train_loss"] = train_loss
            self.generation[i]["train_acc"] = train_acc
    

    def validate_generation(self, X_val, y_val):
        for i in range(self.n_individuals):
            model = self.generation[i]["model"]
            batch_size = self.generation[i]["batch_size"]
            
            # Create a DataLoader with the architecture-specific batch size
            val_loader = create_dataloaders(X=X_val, y=y_val, batch_size=batch_size)
            
            val_loss, val_acc = model.evaluate(val_loader)
            self.generation[i]["val_loss"] = val_loss
            self.generation[i]["val_acc"] = val_acc


    def get_worst_individuals(self, 
                              percentile_drop=15):
    
        n_worst_individuals = max(1, int(self.n_individuals * percentile_drop / 100))  # Ensure at least 1

        # Sort individuals by validation loss in descending order (higher loss is worse)
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["val_loss"], reverse=True) #? Should the criterion be val_loss or val_acc?

        # Extract the keys of the worst individuals
        self.worst_individuals = [key for key, _ in sorted_generation[:n_worst_individuals]]

        

    def drop_worst_individuals(self):
        # Clean up GPU memory before removing references
        for idx in self.worst_individuals:
            if hasattr(self.generation[idx]["model"], "cpu"):
                self.generation[idx]["model"] = self.generation[idx]["model"].cpu()
            # Force garbage collection for the model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Remove worst individuals
        for idx in self.worst_individuals:
            del self.generation[idx]
        
        # Re-index the remaining individuals to maintain continuous keys
        self.generation = {new_idx: val for new_idx, (_, val) in enumerate(self.generation.items())}
        self.n_individuals = len(self.generation)  # Update the count

    def drop_all_except_best(self):
        # Sort individuals by validation loss in ascending order (lower loss is better)
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["val_loss"])
        
        # Keep only the best individual
        best_individual = sorted_generation[0][0]
        best_model_data = self.generation[best_individual]
        
        # Clean up GPU memory for models that will be discarded
        for idx, data in self.generation.items():
            if idx != best_individual:
                if hasattr(data["model"], "cpu"):
                    data["model"] = data["model"].cpu()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.generation = {0: best_model_data}
        self.n_individuals = 1

    def train_best_individual(self, X_train, y_train, num_epochs=1):
        best_model = self.generation[0]["model"]
        batch_size = self.generation[0]["batch_size"]
        # Create a DataLoader with the architecture-specific batch size
        train_loader = create_dataloaders(X=X_train, y=y_train, batch_size=batch_size)
        best_model.oe_train(train_loader, num_epochs=num_epochs)
    
    def return_df(self):
        # As a dataframe
        architectures = []
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        batch_sizes = []
        for i in range(self.n_individuals):
            architectures.append(self.generation[i]["architecture"])
            train_losses.append(self.generation[i]["train_loss"])
            train_accs.append(self.generation[i]["train_acc"])
            val_losses.append(self.generation[i]["val_loss"])
            val_accs.append(self.generation[i]["val_acc"])
            batch_sizes.append(self.generation[i]["batch_size"])        

        # Create a DataFrame with the architectures and their corresponding metrics
        architectures_df = pd.DataFrame(architectures)
        architectures_df['train_loss'] = train_losses
        architectures_df['train_acc'] = train_accs
        architectures_df['val_loss'] = val_losses
        architectures_df['val_acc'] = val_accs
        architectures_df['batch_size'] = batch_sizes

        df = pd.DataFrame(architectures_df)
        return df.sort_values('val_acc', ascending=False).reset_index(drop=True)

    def run_generation(self,
                       X_train, y_train, X_val, y_val,
                       percentile_drop=25):
    
        # Generation is trained, and dropped
        self.train_generation(X_train, y_train, num_epochs=1)
        self.validate_generation(X_val, y_val)
        self.get_worst_individuals(percentile_drop)
        self.drop_worst_individuals()

        return self.generation
    
    def run_ebe(self, n_epochs,
                X_train, y_train, X_val, y_val,
                percentile_drop=25):
        
        for n_epoch in range(n_epochs):
            print(f"Epoch {n_epoch+1}/{n_epochs}")
            self.generation = self.run_generation(X_train, y_train, X_val, y_val, percentile_drop=percentile_drop)
            self.num_models = len(self.generation)
            percentile_drop += 5


# region Functions

def create_dataloaders(X, y, 
                       batch_size, return_as='loaders'):

    # Create DataLoaders
    dataset, dataloader = load_data.create_dataset_and_loader(X, y,
    batch_size=batch_size)
    if return_as == 'loaders':
        return dataloader
    else: 
        return dataset

import ast

# def train_n_models(n_models, architectures_df, input_size, output_size,
#                    X_train, y_train, X_val, y_val):
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     N_MODELS = n_models
#     best_models = architectures_df.head(N_MODELS).reset_index(drop=True)

#     reborn_models = {}
#     results = []
#     for index, model in best_models.iterrows(): 
#         print('--Testing model:', index + 1, '/', N_MODELS)
#         # Extract attributes
#         hidden_layers = ast.literal_eval(model['hidden_layers'])
#         activation_fn_str = model['activation_fn'].split("'")[1].split('.')[-1]
#         if activation_fn_str == 'Sigmoid':
#             activation_fn = nn.Sigmoid
#         elif activation_fn_str == 'ReLU':
#             activation_fn = nn.ReLU
#         elif activation_fn_str == 'Tanh':
#             activation_fn = nn.Tanh
#         elif activation_fn_str == 'LeakyReLU':
#             activation_fn = nn.LeakyReLU

#         dropout_rate = model['dropout_rate']
#         lr = model['learning_rate']
#         batch_size = model['batch_size']
#         optimizer_str = model['optimizer_type'].split("'")[1].split('.')[-1]
#         if optimizer_str == 'Adam':
#             optimizer_type = optim.Adam
#         elif optimizer_str == 'SGD':
#             optimizer_type = optim.SGD

#         # Architectures are reborn
#         reborn_models[index] = DynamicNN(input_size, output_size, 
#                     hidden_layers, 
#                     activation_fn, dropout_rate,
#                     lr, optimizer_type).to(device)
        
#         train_loader = create_dataloaders(X=X_train, y=y_train, batch_size=batch_size)
#         val_loader = create_dataloaders(X=X_val, y=y_val, batch_size=batch_size)
        
#         # Training time
#         train_loss, train_acc, val_loss, val_acc = reborn_models[index].es_train(train_loader, val_loader, es_patience=50, verbose=True, max_epochs=1000)
#                 # Append results
#         results.append({
#             'index': index,
#             'train_loss': train_loss,
#             'train_acc': train_acc,
#             'val_loss': val_loss,
#             'val_acc': val_acc,
#             'hidden_layers': hidden_layers,
#             'activation_fn': activation_fn.__class__.__name__,
#             'dropout_rate': dropout_rate,
#             'learning_rate': lr,
#             'batch_size': batch_size,
#             'optimizer_type': optimizer_type.__name__,
#         })
#     # Convert to DataFrame
#     results_df = pd.DataFrame(results)

#     return results_df

import torch
import torch.nn as nn
import torch.optim as optim
import ast

def create_model_from_row(row, input_size, output_size):
    import ast
    import torch.nn as nn
    import torch.optim as optim

    # Hidden layers
    hidden_layers = row.get('hidden_layers', [128, 64])
    if isinstance(hidden_layers, str):
        hidden_layers = ast.literal_eval(hidden_layers)

    # Activation function
    activation_raw = row.get('activation_fn', nn.ReLU)
    if isinstance(activation_raw, str):
        activation_name = activation_raw
    elif hasattr(activation_raw, '__name__'):
        activation_name = activation_raw.__name__
    else:
        activation_name = 'ReLU'

    activation_map = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
    }
    activation_fn = activation_map.get(activation_name, nn.ReLU)

    # Dropout
    dropout_rate = row.get('dropout_rate', 0.2)

    # Optimizer and learning rate
    lr = row.get('lr', 0.001)
    optimizer_type = row.get('optimizer_type', 'adam')
    weight_decay = row.get('weight_decay', 0.0)
    momentum = row.get('momentum', None)

    # Skip connections
    use_skip = row.get('use_skip_connections', False)

    # Initializer
    initializer = row.get('initializer', 'xavier_uniform')

    # LR Scheduler
    lr_scheduler = row.get('lr_scheduler', 'none')
    scheduler_params = row.get('scheduler_params', {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = DynamicNN(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activation_fn=activation_fn,
        dropout_rate=dropout_rate,
        lr=lr,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        momentum=momentum,
        use_skip_connections=use_skip,
        initializer=initializer,
        lr_scheduler=lr_scheduler,
        scheduler_params=scheduler_params,
        device=device
    ).to(device)

    return model
#endregion