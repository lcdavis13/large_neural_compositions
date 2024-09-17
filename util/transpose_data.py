import pandas as pd

fname = "../data/Ptrain.csv"
data = pd.read_csv(fname, header=None)
print(data.shape)

transposed_data = data.transpose()
print(transposed_data.shape)

transposed_data.to_csv(fname, header=False, index=False)

print("Done")
