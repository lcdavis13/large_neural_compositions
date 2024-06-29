import math
import csv
import time

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from sklearn.model_selection import KFold
# from torchdiffeq import odeint_adjoint as odeint  # tiny memory footprint but it is intractible for large models such as cNODE2 with Waimea data



def loss_bc(x, y):  # Bray-Curtis Dissimilarity
    return torch.sum(torch.abs(x - y)) / torch.sum(torch.abs(x + y))   # DKI implementation
    # return torch.sum(torch.abs(x - y)) / (2.0 * x.shape[0])   # simplified by assuming every vector is a species composition that sums to 1



def process_data(y):
    # produces X (assemblage) from Y (composition), normalizes the composition to sum to 1, and transposes the data
    x = y.copy()
    x[x > 0] = 1
    y = y / y.sum(axis=0)[np.newaxis, :]
    x = x / x.sum(axis=0)[np.newaxis, :]
    y = y.astype(np.float32)
    x = x.astype(np.float32)
    y = torch.from_numpy(y.T)
    x = torch.from_numpy(x.T)
    return x, y


def load_data(filepath_train):
    # Load data
    y = np.loadtxt(filepath_train, delimiter=',')
    x, y = process_data(y)  # Assuming process_data is defined somewhere
    
    # Move data to device if specified
    if device:
        x = x.to(device)
        y = y.to(device)
    
    return x, y


def fold_data(x, y, k=5):
    # Split data into k folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_data = []
    
    for train_index, valid_index in kf.split(x):
        x_train, x_valid = x[train_index], x[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        fold_data.append((x_train, y_train, x_valid, y_valid))
    
    return fold_data


def get_batch(x, y, mb_size, current_index):
    end_index = current_index + mb_size
    if end_index > x.size(0):
        end_index = x.size(0)
    batch_indices = torch.arange(current_index, end_index, dtype=torch.long).to(device)
    x_batch = x[batch_indices, :]
    y_batch = y[batch_indices, :]
    # print(f'x {x_batch.shape}')
    # print(f'x {x_batch[0, :]}')
    # print(f'y {y_batch.shape}')
    # print(f'y {y_batch[0, :]}')
    return x_batch, y_batch, end_index


# class ODEFunc_cNODE2_DKI_unbatched(nn.Module):  # original DKI implementation of cNODE2, but will crash if you send batched data
#     def __init__(self, N):
#         super(ODEFunc_cNODE2_DKI_unbatched, self).__init__()
#         self.fcc1 = nn.Linear(N, N)
#         self.fcc2 = nn.Linear(N, N)
#
#     def forward(self, t, y):
#         out = self.fcc1(y)
#         out = self.fcc2(out)
#         f = torch.matmul(torch.matmul(torch.ones(y.size(dim=1), 1).to(device), y), torch.transpose(out, 0, 1))
#         return torch.mul(y, out - torch.transpose(f, 0, 1))
#
# class ODEFunc_cNODE2_DKI(nn.Module): # DKI implementation of cNODE2 modified to allow batches
#     def __init__(self, N):
#         super(ODEFunc_cNODE2_DKI, self).__init__()
#         self.fcc1 = nn.Linear(N, N)
#         self.fcc2 = nn.Linear(N, N)
#
#     def forward(self, t, y):
#         y = y.unsqueeze(1)  # B x 1 x N
#         out = self.fcc1(y)
#         out = self.fcc2(out)
#         f = torch.matmul(torch.matmul(torch.ones(y.size(dim=-1), 1).to(device), y), torch.transpose(out, -2, -1))
#         dydt = torch.mul(y, out - torch.transpose(f, -2, -1))
#         return dydt.squeeze(1)  # B x N
#
# class cNODE2_DKI(nn.Module):
#     def __init__(self, N):
#         super(cNODE2_DKI, self).__init__()
#         self.func = ODEFunc_cNODE2_DKI(N)
#
#     def forward(self, t, x):
#         x = odeint(self.func, x, t)[-1]
#         return x

class ODEFunc_cNODE2(nn.Module): # optimized implementation of cNODE2
    def __init__(self, N):
        super(ODEFunc_cNODE2, self).__init__()
        self.fcc1 = nn.Linear(N, N)
        # self.bn1 = nn.BatchNorm1d(N)
        self.fcc2 = nn.Linear(N, N)

    def forward(self, t, x):
        fx = self.fcc1(x)  # B x N
        # fx = self.bn1(fx)
        fx = self.fcc2(fx)  # B x N

        xT_fx = torch.sum(x*fx, dim=-1).unsqueeze(1) # B x 1 (batched dot product)
        diff = fx - xT_fx # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt # B x N
class cNODE2(nn.Module):
    def __init__(self, N):
        super(cNODE2, self).__init__()
        self.func = ODEFunc_cNODE2(N)
    
    def forward(self, t, x):
        x = odeint(self.func, x, t)[-1]
        return x

class Embedded_cNODE2(nn.Module):
    def __init__(self, N, M):
        super(Embedded_cNODE2, self).__init__()
        self.embed = nn.Linear(N, M)  # can't use a proper embedding matrix because there are multiple active channels, not one-hot encoded
        # self.softmax = nn.Softmax(dim=-1)
        self.func = ODEFunc_cNODE2(M)
        self.unembed = nn.Linear(M, N)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, t, x):
        x = self.embed(x)
        x = self.softmax(x)  # the ODE expects a few-hot encoded species assemblage summing to 1. This just doesn't make much sense to connect to it. We should instead use two channels - embed IDs for each species, and a small dense list of their abundances.
        x = odeint(self.func, x, t)[-1]
        x = self.unembed(x)
        x = self.softmax(x)  # TODO: If I don't have softmax, the outputs aren't normalized resulting in absurdly high loss. If I do have softmax, I get underflow errors. And when I get lucky and have no underflow, the model doesn't learn at all.
        return x


def ceildiv(a, b):
    return -(a // -b)


def stream_results(filename, print_console, *args, prefix="", suffix=""):
    if len(args) % 2 != 0:
        raise ValueError("Arguments should be in pairs of names and values.")
    
    names = args[0::2]
    values = args[1::2]
    
    if print_console:
        print(prefix + (", ".join([f"{name}: {value}" for name, value in zip(names, values)])) + suffix)
    
    # Check if file exists
    if filename:
        # Initialize the set of filenames if it doesn't exist
        if not hasattr(stream_results, 'filenames'):
            stream_results.filenames = set()
        
        # Check if it's the first time the function is called for this filename
        if filename not in stream_results.filenames:
            stream_results.filenames.add(filename)
            file_started = False
        else:
            file_started = True
        
        mode = 'a' if file_started else 'w'
        with open(filename, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # If the file is new, write the header row
            if not file_started:
                writer.writerow(names)
            
            # Write the values row
            writer.writerow(values)


def validate_epoch(model, x_val, y_val, minibatch_examples, t):
    model.eval()
    
    total_loss = 0
    total_samples = x_val.size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0  # Initialize current index to start of dataset

    for mb in range(minibatches):
        x, y, current_index = get_batch(x_val, y_val, minibatch_examples, current_index)
        mb_examples = x.size(0)

        with torch.no_grad():
            y_pred = model(t, x).to(device)

            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * mb_examples  # Multiply loss by batch size

    avg_loss = total_loss / total_samples
    return avg_loss



def train_epoch(model, x_train, y_train, minibatch_examples, accumulated_minibatches, optimizer, t, outputs_per_epoch, prev_examples, fold, epoch_num, filepath_out_incremental):
    model.train()
    
    total_loss = 0
    total_samples = x_train.size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0  # Initialize current index to start of dataset
    new_examples = 0

    stream_interval = max(1, minibatches // outputs_per_epoch)

    optimizer.zero_grad()

    # set up metrics for streaming
    prev_time = time.time()
    stream_loss = 0
    stream_examples = 0

    # TODO: shuffle the data before starting an epoch
    for mb in range(minibatches):

        x, y, current_index = get_batch(x_train, y_train, minibatch_examples, current_index) #
        mb_examples = x.size(0)
        
        if current_index >= total_samples:
            current_index = 0  # Reset index if end of dataset is reached
        
        y_pred = model(t, x).to(device)

        loss = loss_fn(y_pred, y)
        actual_loss = loss.item() * mb_examples
        loss = loss / accumulated_minibatches # Normalize the loss by the number of accumulated minibatches, since loss function can't normalize by this

        loss.backward()
        
        #del y_pred, loss

        total_loss += actual_loss
        new_examples += mb_examples
        
        stream_loss += actual_loss
        stream_examples += mb_examples

        if (mb + 1) % stream_interval == 0:
            end_time = time.time()
            examples_per_second = stream_examples / (end_time - prev_time)
            stream_results(filepath_out_incremental, True,
               "fold", fold,
                "epoch", epoch_num,
                "minibatch", mb,
                "total examples seen", prev_examples + new_examples,
                "Avg Loss", stream_loss / stream_examples,
                "Examples per second", examples_per_second)
            stream_loss = 0
            prev_time = end_time
            stream_examples = 0

        if ((mb + 1) % accumulated_minibatches == 0) or (mb == minibatches - 1):
            optimizer.step()
            optimizer.zero_grad()

        #del x, y


    avg_loss = total_loss / total_samples
    new_total_examples = prev_examples + new_examples
    return avg_loss, new_total_examples


def run_epochs(model, max_epochs, minibatch_examples, accumulated_minibatches, LR, x_train, y_train, x_valid, y_valid, t,
               filepath_out_incremental, filepath_out_epoch, filepath_out_model, fold, earlystop_patience=10, outputs_per_epoch=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    best_loss = float('inf')
    best_epoch = -1
    epochs_worsened = 0
    early_stop = False

    train_examples_seen = 0
    start_time = time.time()
    
    # print parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in model: {num_params}")
    
    # initial validation benchmark
    l_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, t)
    stream_results(filepath_out_epoch, True,
        "fold", fold,
        "epoch", -1,
        "training examples", 0,
        "Training Loss", -1.0,
        "Validation Loss", l_val,
        "Elapsed Time", -1.0,
        "GPU Footprint (MB)", -1.0,
        prefix="================PRE-VALIDATION===============\n",
        suffix="\n=============================================\n")
    
    
    for e in range(max_epochs):
        l_trn, train_examples_seen = train_epoch(model, x_train, y_train, minibatch_examples, accumulated_minibatches,
                                                 optimizer, t, outputs_per_epoch, train_examples_seen, fold, e,
                                                 filepath_out_incremental)
        l_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, t)
        # l_trn = validate_epoch(model, x_train, y_train, minibatch_examples, t) # Sanity test, should use running loss from train_epoch instead as a cheap approximation
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        gpu_memory_reserved = torch.cuda.memory_reserved(device)
        
        stream_results(filepath_out_epoch, True,
            "fold", fold,
            "epoch", e,
            "training examples", train_examples_seen,
            "Training Loss", l_trn,
            "Validation Loss", l_val,
            "Elapsed Time", elapsed_time,
            "GPU Footprint (MB)", gpu_memory_reserved / (1024 ** 2),
            prefix="================PRE-VALIDATION===============\n",
            suffix="\n=============================================\n")
        
        # early stopping & model backups
        if l_val < best_loss:
            best_loss = l_val
            best_epoch = e
            epochs_worsened = 0
            torch.save(model.state_dict(), filepath_out_model)
        else:
            epochs_worsened += 1
            print(f"VALIDATION DIDN'T IMPROVE. PATIENCE {epochs_worsened}/{earlystop_patience}")
            # early stop
            if epochs_worsened >= earlystop_patience:
                print(f'Early stopping triggered after {e + 1} epochs.')
                early_stop = True
                break
    
    if not early_stop:
        print(f"Completed training. Optimal loss was {best_loss}")
    else:
        print(f"Training stopped early due to lack of improvement in validation loss. Optimal loss was {best_loss}")
    
    # TODO: Check if this is the best model of a given name, and if so, save the weights and logs to a separate folder for that model name
    
    return best_loss, best_epoch


# data paths

# dataname = "waimea"
# dataname = "waimea_condensed"
dataname = "dki"

filepath_train = f'data/{dataname}_train.csv'
filepath_test = f'data/{dataname}_test.csv'

filepath_out_incremental = f'results/{dataname}_incremental.csv'
filepath_out_epoch = f'results/{dataname}_epochs.csv'
filepath_out_fold = f'results/{dataname}_folds.csv'
filepath_out_model = f'results/{dataname}_model.pth'

kfolds = 5



# hyperparameters

max_epochs = 50000
minibatch_examples = 500
accumulated_minibatches = 1
earlystop_patience = 10

loss_fn = loss_bc

LR_base = 0.002
LR = LR_base*math.sqrt(minibatch_examples * accumulated_minibatches)
weight_decay = 0.0003



# main
if __name__ == "__main__":
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # load data
    x,y = load_data(filepath_train)
    data_folded = fold_data(x, y, kfolds)
    
    _, N = x.shape
    
    model = cNODE2(N).to(device)
    # model = Embedded_cNODE2(N, math.isqrt(N)).to(device)
    
    print('dataset:', filepath_train)
    print(f'training data shape: {data_folded[0][0].shape}')
    print(f'validation data shape: {data_folded[0][1].shape}')
    
    # time step "data"
    ode_timesteps = 2 # must be at least 2
    timesteps = torch.arange(0.0, 1.0, 1.0 / ode_timesteps).to(device)
    
    fold_losses = []
    
    for fold_num, data_fold in enumerate(data_folded):
        x_train, y_train, x_valid, y_valid = data_fold
        print(f"Fold {fold_num + 1}/{kfolds}")
    
        val, e = run_epochs(model, max_epochs, minibatch_examples, accumulated_minibatches, LR, x_train, y_train, x_valid, y_valid, timesteps, filepath_out_incremental, filepath_out_epoch, filepath_out_model, fold_num, earlystop_patience=earlystop_patience, outputs_per_epoch=400)
        fold_losses.append(val)
        
        stream_results(filepath_out_fold, True,
            "fold", fold_num,
            "Validation Loss", val,
            "epochs", e,
            prefix="\n========================================FOLD=========================================\n",
            suffix="\n=====================================================================================\n")
    
    # min of mean and mode to avoid over-optimism from outliers
    model_score = np.min([np.mean(fold_losses), np.median(fold_losses)])
    print(f"Losses: {fold_losses}")
    print(f"Model score: {model_score}")