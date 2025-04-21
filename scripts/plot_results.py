import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load the CSV
df = pd.read_csv("exp_results_full.csv")

# 2) Identify and sort the unique noise‐factor levels
filter_factors = np.array(sorted(df["filter_factor"].unique()))
data_factors   = sorted(df["data_factor"].unique())

# 3) Pivot into tables:
#     - mean_pe.loc[data][filter] = mean position error
#     - std_pe .loc[data][filter] = its std
#     - likewise for ANEES
mean_pe    = df.pivot(index="data_factor",   columns="filter_factor", values="mean_mpe")
std_pe     = df.pivot(index="data_factor",   columns="filter_factor", values="std_mpe")
mean_anees = df.pivot(index="data_factor",   columns="filter_factor", values="mean_anees")
std_anees  = df.pivot(index="data_factor",   columns="filter_factor", values="std_anees")

# 4) Build the error‐bar plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for dfactor in data_factors:
    ax1.errorbar(
        filter_factors,
        mean_pe.loc[dfactor].values,
        yerr=std_pe.loc[dfactor].values,
        marker='o',
        label=f"data-noise={dfactor:g}"
    )
    ax2.errorbar(
        filter_factors,
        mean_anees.loc[dfactor].values,
        yerr=std_anees.loc[dfactor].values,
        marker='s',
        label=f"data-noise={dfactor:g}"
    )

# 5) Style and save
for ax, ylabel in ((ax1, "Mean Position Error"), (ax2, "Mean ANEES")):
    ax.set_xscale("log")
    ax.set_xlabel("Filter noise factor")
    ax.set_ylabel(ylabel)
    ax.legend(title="Data noise factor")
    ax.grid(True)

plt.tight_layout()
plt.savefig("errorbar_slice_plots.png")
print("Saved errorbar_slice_plots.png")