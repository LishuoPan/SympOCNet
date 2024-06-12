import pandas as pd
import matplotlib.pyplot as plt

unified = True

with open("2024-06-11_13-46-44/2024-06-11_13-46-44.log", "r") as f:
    lines = f.readlines()

data = []
for line in lines:
    line = line.strip().split("-")[-1]
    if line.startswith(" loss_"):
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
