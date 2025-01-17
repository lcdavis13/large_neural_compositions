# This file is based on the DKI paper repo, specifically: https://github.com/spxuw/DKI/blob/main/Simulated_data_generation.R
import json

import numpy as np
import pandas as pd
import os
from scipy.integrate import odeint  # could also use torchdiffeq.odeint but we don't need gradients here
import scipy.sparse as sp
from dotsy import dicy


def glv_ode(x, t, A, r):
    """
    x: current population sizes
    t: time (required by odeint but not used here)
    A: interaction matrix
    r: intrinsic growth rates
    """
    # dydt = x * (r + np.dot(A, x))
    
    # Ensure x and r are sparse column vectors
    if sp.issparse(A):
        if not sp.issparse(x):
            x = sp.csr_matrix(x).T  # Convert x to a sparse column vector
        if not sp.issparse(r):
            r = sp.csr_matrix(r).T  # Convert r to a sparse column vector
    
    #
    # print(type(x), type(r), type(A))
    # print(x.shape, r.shape, A.shape)
    
    # Element-wise multiplication between sparse vectors is done using `.multiply()`
    growth_term = x.multiply(r)  # Element-wise multiplication between sparse vectors
    interaction_term = x.multiply(A.dot(x))  # Matrix-vector product (sparse dot)
    
    # Add the growth and interaction terms
    dydt = growth_term + interaction_term
    
    # Convert to a dense array to apply the condition
    dydt = dydt.toarray().flatten()
    
    return dydt


def glv(N, A, r, x_0, tstart, tend, tstep):
    """
    N: number of species
    A: interaction matrix
    r: intrinsic growth rates
    x_0: initial populations
    tstart: start time
    tend: end time
    tstep: time step size
    """
    
    # Create time points
    t = np.arange(tstart, tend + tstep, tstep)
    
    # Solve ODEs
    result = odeint(glv_ode, x_0, t, args=(A, r))
    
    return result[-1]


def generate_composition_data(N, M, A, r, mean_richness, stdev_richness, C, sigma, nu, boost_rate, path_dir, resume, batch_size=100):

    steady_state_file = f'{path_dir}/Ptrain.csv'

    steady_state_absolute = sp.lil_matrix((M, N))  # Change the shape to (M, N), rows = samples, cols = data points
    steady_state_relative = sp.lil_matrix((M, N))  # Same here

    update_interval = batch_size  # max(M // 100, 1)

    start_sample = 0

    if resume and os.path.exists(steady_state_file):
        print("Loading existing composition data...")
        existing_data = pd.read_csv(steady_state_file, header=None).values
        loaded_samples = existing_data.shape[0]  # Load based on rows
        steady_state_relative[:loaded_samples, :] = existing_data  # Populate loaded rows
        start_sample = loaded_samples

        if start_sample >= M:
            return steady_state_absolute.tocsr(), steady_state_relative.tocsr()

    print(f"Generating new composition data from sample {start_sample + 1}/{M}...")

    batch_data = []  # Buffer to hold batch data

    for i in range(start_sample, M):
        if i % update_interval == 0:
            print(f"Processing sample {i + 1}/{M}")

        np.random.seed(i)
        richness = int(np.random.normal(mean_richness, stdev_richness))
        richness = max(2, min(N - 1, richness))  # Ensure that richness is within [2, N-1]
        collection = np.random.choice(np.arange(N), richness, replace=False)
        y_0 = np.zeros(N)
        y_0[collection] = np.random.uniform(size=richness)
        x = glv(N, A, r, y_0, 0, 100, 100)

        steady_state_absolute[i, :] = x[:]
        steady_state_relative[i, :] = x[:] / np.sum(x[:]) if np.sum(x[:]) != 0 else x[:]

        # Ensure that the values are non-negative
        row_data = np.maximum(steady_state_relative[i, :].toarray(), 0)

        # Add the row data to the batch buffer
        batch_data.append(row_data.flatten())

        # Write the batch to the file when the batch size is reached
        if i % batch_size == 0:
            pd.DataFrame(batch_data).to_csv(steady_state_file, mode='a', index=False, header=False)
            batch_data = []  # Clear the batch buffer

    # Write any remaining data in the buffer to the file
    if batch_data:
        pd.DataFrame(batch_data).to_csv(steady_state_file, mode='a', index=False, header=False)

    return steady_state_absolute.tocsr(), steady_state_relative.tocsr()



def generate_ecosystem(N, C, sigma, nu, boost_rate, path_dir, resume):
    A_file = f'{path_dir}/A.csv'
    r_file = f'{path_dir}/r.csv'
    
    if resume and os.path.exists(A_file) and os.path.exists(r_file):
        print("Loading existing ecosystem data...")
        A = sp.csr_matrix(pd.read_csv(A_file, header=None).values)
        r = pd.read_csv(r_file, header=None).values.flatten()
    else:
        print("Generating new ecosystem data...")
        np.random.seed(234)
        A = sp.random(N, N, density=C, data_rvs=lambda size: np.random.normal(0, sigma, size)).tocsr()
        
        non_zero_indices = A.nonzero()
        num_to_boost = int(boost_rate * len(non_zero_indices[0]))
        
        if num_to_boost > 0:
            boost_indices = np.random.choice(len(non_zero_indices[0]), size=num_to_boost, replace=False)
            row_indices = non_zero_indices[0][boost_indices]
            col_indices = non_zero_indices[1][boost_indices]
            boosts = np.random.lognormal(mean=0, sigma=nu, size=num_to_boost)
            
            for i in range(num_to_boost):
                A[row_indices[i], col_indices[i]] *= boosts[i]
        
        A.setdiag(-1)
        r = np.random.uniform(size=N)
        
        pd.DataFrame(A.toarray()).to_csv(A_file, index=False, header=False)
        pd.DataFrame(r).to_csv(r_file, index=False, header=False)
    
    return A, r


def generate_keystoneness_data(M, N, A, r, steady_state_absolute, path_dir):
    # Generate test samples for keystone species calculation
    species_id = []
    sample_id = []
    absent_composition = []
    absent_collection = []
    
    for j1 in range(N):
        print(f"Processing species {j1 + 1}/{N}")
        for j2 in range(M):
            if steady_state_absolute[j1, j2] > 0:
                y_0 = steady_state_absolute[:, j2].copy()
                y_0_binary = (y_0 > 0).astype(int)
                
                if np.sum(y_0_binary) > 1:
                    y_0[j1] = 0
                    x = glv(N, A, r, y_0, 0, 100, 0.1)
                    absent_composition.append(x[:] / np.sum(x[:]))
                    species_id.append(j1)
                    sample_id.append(j2)
                    y_0[y_0 > 0] = 1
                    absent_collection.append(y_0 / np.sum(y_0))
    
    # Save results to CSV files
    pd.DataFrame(species_id).to_csv(f'{path_dir}/Species_id.csv', index=False, header=False)
    pd.DataFrame(sample_id).to_csv(f'{path_dir}/Sample_id.csv', index=False, header=False)
    pd.DataFrame(absent_composition).to_csv(f'{path_dir}/Ptest.csv', index=False, header=False)
    pd.DataFrame(absent_collection).to_csv(f'{path_dir}/Ztest.csv', index=False, header=False)


def write(obj, filename):
    with open(filename, 'w') as outp:
        json.dump(obj, outp, indent=4)
        
        
def read(filename):
    with open(filename, 'r') as inp:
        return dicy(json.load(inp))


def params_match(p, path_params):
    if not os.path.exists(path_params):
        return False
    
    p2 = read(path_params)
    p2.M = p.M # We want to be able to resume computing with a different number of samples
    
    return p == p2


def main():
    restart_computation = False
    
    path_prefix = './data/synth/gLV_'
    
    p = dicy()
    p.N = 69  # number of species
    p.M = 10000  # number of samples
    
    p.mean_richness = 50  # richness (proportion of total species present in each sample) (1/36 ~= 2.77% for Waimea)
    p.stdev_richness = 8.42  # standard deviation of richness

    p.C = 0.05  # connectivity rate
    p.sigma = 0.01  # characteristic interaction strength
    
    p.nu = 1.0  # boosting strength
    p.boost_rate = 1.0  # boosting rate. In the original paper's repo it was hardcoded to 1, in which case the random choice effectively does nothing. Rather than removing that unused feature, I've parameterized it.
    
    path_dir = f'{path_prefix}_N{p.N}_C{p.C}_nu{p.nu}'
    os.makedirs(path_dir, exist_ok=True)
    
    path_params = f'{path_dir}/params.json'
    
    resume = (not restart_computation) and params_match(p, path_params)
    
    if not resume:
        write(p, path_params)
    else:
        print("Resuming computation")

    A, r = generate_ecosystem(p.N, p.C, p.sigma, p.nu, p.boost_rate, path_dir, resume)
    
    steady_state_absolute, steady_state_relative = generate_composition_data(p.N, p.M, A, r, p.mean_richness, p.stdev_richness, p.C, p.sigma, p.nu, p.boost_rate, path_dir, resume)
    
    # generate_keystoneness_data(p.M, p.N, A, r, steady_state_absolute, path_dir)
    


# TODO: Prune mutualistic interactions, since they cause gLV models to becaome unstable. Can maybe get away with pruning only strong symmetric mutualisms, α_ij*α_ji >= 1


if __name__ == "__main__":
    main()
