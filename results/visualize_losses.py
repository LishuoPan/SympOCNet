import pandas as pd
import matplotlib.pyplot as plt

unified = True

with open("parameters_nn_added/outputs/new_losses.log", "r") as f:
    lines = f.readlines()

data = []
for line in lines:
    line = line.strip().split("INFO:root:")[1]
    items = line.split(", ")
    row = {}
    for item in items:
        key, value = item.split(": ")
        row[key] = float(value)
    data.append(row)

df = pd.DataFrame(data)
df = df.iloc[::100, :]
if not unified:
    fig, axs = plt.subplots(len(df.columns), 1)

for i, column in enumerate(df.columns):
    if unified:
        plt.plot(df[column], label=f"{column}")
        plt.legend()
    else:
        axs[i].plot(df[column], label=f"{column}")
        axs[i].legend()

plt.yscale("log")
plt.tight_layout()
plt.show()
