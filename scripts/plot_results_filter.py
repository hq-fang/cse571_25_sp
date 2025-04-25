#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1) Load the CSV
    df = pd.read_csv("../results/exp_results_filter.csv")
    # 2) Keep only data_factor == 1 and sort by filter_factor
    df = df[df["data_factor"] == 1].sort_values("filter_factor")

    # 3) Extract arrays
    x           = df["filter_factor"].values
    mean_pe     = df["mean_mpe"].values
    std_pe      = df["std_mpe"].values
    mean_anees  = df["mean_anees"].values
    std_anees   = df["std_anees"].values

    # 4) Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mean Position Error
    ax1.errorbar(x, mean_pe, yerr=std_pe, marker='o', capsize=5)
    ax1.set_xscale('log')
    ax1.set_xlabel("Filter noise factor")
    ax1.set_ylabel("Mean Position Error")
    ax1.set_title("Mean Position Error vs Filter Noise")
    ax1.grid(True)

    # Mean ANEES
    ax2.errorbar(x, mean_anees, yerr=std_anees, marker='s', capsize=5)
    ax2.set_xscale('log')
    ax2.set_xlabel("Filter noise factor")
    ax2.set_ylabel("Mean ANEES")
    ax2.set_title("Mean ANEES vs Filter Noise")
    ax2.grid(True)

    plt.tight_layout()
    # Save and show
    plt.savefig("filter_noise_slice_plots.png")
    print("Saved")
    plt.show()

if __name__ == "__main__":
    main()