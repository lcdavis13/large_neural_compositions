import math
import csv
import time

import torch
import numpy as np
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint  # tiny memory footprint but it is intractible for large models such as cNODE2 with Waimea data



def loss_bc(x, y):  # Bray-Curtis Dissimilarity, but simplified by assuming every vector is a species composition that sums to 1
    return torch.sum(torch.abs(x - y)) / (2.0 * x.shape[0])
    # return torch.sum(torch.abs(x - y)) / torch.sum(torch.abs(x + y))



def process_data(P):
    Z = P.copy()
    Z[Z > 0] = 1
    P = P / P.sum(axis=0)[np.newaxis, :]
    Z = Z / Z.sum(axis=0)[np.newaxis, :]
    P = P.astype(np.float32)
    Z = Z.astype(np.float32)
    P = torch.from_numpy(P.T)
    Z = torch.from_numpy(Z.T)
    return Z, P


def load_train_valid_data(filepath_train, valid_ratio=0.2):
    global P_val, P_train, x_train, y_train, x_valid, y_valid, N
    # load data
    P = np.loadtxt(filepath_train, delimiter=',')
    number_of_cols = P.shape[1]
    random_indices = np.random.choice(number_of_cols, size=int(valid_ratio * number_of_cols), replace=False)
    P_val = P[:, random_indices]
    P_train = P[:, np.setdiff1d(range(0, number_of_cols), random_indices)]
    x_train, y_train = process_data(P_train)
    x_valid, y_valid = process_data(P_val)
    M, N = x_train.shape
    # Move data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)
    
    return x_train, y_train, x_valid, y_valid


# class ODEFunc_cNODE2(torch.nn.Module):
#     def __init__(self, N):
#         super(ODEFunc_cNODE2, self).__init__()
#         self.fcc1 = torch.nn.Linear(N, N)
#         self.fcc2 = torch.nn.Linear(N, N)
#
#     def forward(self, t, y):
#         out = self.fcc1(y)
#         out = self.fcc2(out)
#         f = torch.matmul(torch.matmul(torch.ones(y.size(dim=1), 1).to(device), y), torch.transpose(out, 0, 1))
#         return torch.mul(y, out - torch.transpose(f, 0, 1))
#
#
# class ODEFunc_cNODE2_optimized(torch.nn.Module):
#     def __init__(self, N):
#         super(ODEFunc_cNODE2_optimized, self).__init__()
#         self.fcc1 = torch.nn.Linear(N, N)
#         self.fcc2 = torch.nn.Linear(N, N)
#
#     def forward(self, t, y):
#         out = self.fcc1(y)
#         out = self.fcc2(out)
#         f = torch.matmul(torch.ones(y.size(dim=1), 1).to(device), torch.matmul(y, torch.transpose(out, 0, 1)))
#         return torch.mul(y, out - torch.transpose(f, 0, 1))


class ODEFunc_cNODE2_batched(torch.nn.Module):   # THIS IS THE GOOD ONE... OR AT LEAST IT WAS
    def __init__(self, N):
        super(ODEFunc_cNODE2_batched, self).__init__()
        self.fcc1 = torch.nn.Linear(N, N)
        self.fcc2 = torch.nn.Linear(N, N)

    def forward(self, t, x):
        # initially x is B x N
        x = x.unsqueeze(1)  # B x 1 x N

        fx = self.fcc1(x)  # B x 1 x N
        fx = self.fcc2(fx)  # B x 1 x N

        xT_fx = torch.matmul(x, torch.transpose(fx, -2, -1))  # B x 1 x 1   (This is optimized vs DKI's implementation, which creates a NxN matrix instead of 1x1)
        # print(xT_fx.shape)
        ones = torch.ones(x.size(dim=-1), 1).to(device)  # N x 1
        # print(ones.shape)
        ones_xT_fx = torch.matmul(ones, xT_fx)  # B x N x 1
        # print(ones_xT_fx.shape)
        dxdt = torch.mul(x, fx - torch.transpose(ones_xT_fx, -2, -1))  # B x 1 x N
        # print(dxdt.shape)
        return dxdt.squeeze(1)  # B x N


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

def get_batch(x, y, mb_size, current_index):
    end_index = current_index + mb_size
    if end_index > x.size(0):
        end_index = x.size(0)
    batch_indices = torch.arange(current_index, end_index, dtype=torch.long).to(device)
    # print(f'x {x[batch_indices[0], :]}')
    # print(f'y {y[batch_indices[0], :]}')
    return x[batch_indices, :], y[batch_indices, :], end_index


def validate_epoch(x_val, y_val, minibatch_examples, func, t):
    func.eval()
    
    total_loss = 0
    total_samples = x_val.size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0  # Initialize current index to start of dataset

    for mb in range(minibatches):
        x, y, current_index = get_batch(x_val, y_val, minibatch_examples, current_index)
        mb_examples = x.size(0)

        with torch.no_grad():
            y_pred = odeint(func, x, t)[-1].to(device)

            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * mb_examples  # Multiply loss by batch size

    avg_loss = total_loss / total_samples
    return avg_loss



def train_epoch(x_train, y_train, minibatch_examples, accumulated_minibatches, func, optimizer, t, outputs_per_epoch, prev_examples, epoch_num, filepath_out_incremental):
    func.train()
    
    total_loss = 0
    total_samples = x_train.size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0  # Initialize current index to start of dataset
    new_examples = 0

    stream_interval = minibatches // outputs_per_epoch

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

        y_pred = odeint(func, x, t)[-1].to(device)

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


def run_epochs(max_epochs, minibatch_examples, accumulated_minibatches, LR, x_train, y_train, x_valid, y_valid,
               filepath_out_incremental, filepath_out_epoch, filepath_out_model, earlystop_patience=10, outputs_per_epoch=10):
    func = ODEFunc(N).to(device)
    optimizer = torch.optim.Adam(func.parameters(), lr=LR, weight_decay=weight_decay)
    best_loss = float('inf')
    epochs_worsened = 0
    early_stop = False

    train_examples_seen = 0
    prev_time = time.time()
    
    for e in range(max_epochs):
        l_trn, train_examples_seen = train_epoch(x_train, y_train, minibatch_examples, accumulated_minibatches,
                                                 func, optimizer, t, outputs_per_epoch, train_examples_seen, e,
                                                 filepath_out_incremental)
        l_val = validate_epoch(x_valid, y_valid, minibatch_examples, func, t)
        # l_trn = validate_epoch(x, y, minibatch_examples, func, t) # Sanity test, should use running loss from train_epoch instead as a cheap approximation
        
        end_time = time.time()
        elapsed_time = end_time - prev_time
        prev_time = end_time
        
        stream_results(filepath_out_epoch, True,
                       "epoch", e,
                       "training examples", train_examples_seen,
                       "Training Loss", l_trn,
                       "Validation Loss", l_val,
                       "Elapsed Time", elapsed_time,
                       prefix="====================EPOCH====================\n",
                       suffix="\n=============================================")
        
        # early stopping & model backups
        if l_val < best_loss:
            best_loss = l_val
            epochs_worsened = 0
            torch.save(func.state_dict(), filepath_out_model)
        else:
            epochs_worsened += 1
            print(f"VALIDATION DIDN'T IMPROVE. PATIENCE {epochs_worsened}/{earlystop_patience}")
            # early stop
            if epochs_worsened >= earlystop_patience:
                print(f'Early stopping triggered after {e + 1} epochs.')
                early_stop = True
                break
    
    if not early_stop:
        print("Completed training")
    else:
        print("Training stopped early due to lack of improvement in validation loss.")
    
    # TODO: Check if this is the best model yet, and if so, save the weights and logs to a separate folder (and architecture, but how? copy the entire source code?)


# data paths

# dataname = "waimea"
dataname = "dki"

filepath_train = f'data/{dataname}_train.csv'
filepath_test = f'data/{dataname}_test.csv'

filepath_out_epoch = f'results/{dataname}_epochs.csv'
filepath_out_incremental = f'results/{dataname}_incremental.csv'
filepath_out_model = f'results/{dataname}_model.pth'

data_valid_ratio = 0.2



# hyperparameters

max_epochs = 1000
minibatch_examples = 20
accumulated_minibatches = 1

ODEFunc = ODEFunc_cNODE2_batched
loss_fn = loss_bc

LR_base = 0.002
LR = LR_base*math.sqrt(minibatch_examples * accumulated_minibatches)
weight_decay = 0.0003

ode_timesteps = 2 # must be at least 2


# main
if __name__ == "__main__":
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # load data
    x_train, y_train, x_valid, y_valid = load_train_valid_data(filepath_train, data_valid_ratio)
    
    print('dataset:', filepath_train)
    print(f'training data shape: {P_train.shape}')
    print(f'validation data shape: {P_val.shape}')
    
    # time step "data"
    t = torch.arange(0.0, 1.0, 1.0 / ode_timesteps).to(device)
    
    run_epochs(max_epochs, minibatch_examples, accumulated_minibatches, LR, x_train, y_train, x_valid, y_valid, filepath_out_incremental, filepath_out_epoch, filepath_out_model, earlystop_patience=10, outputs_per_epoch=4)
    
    print("done")