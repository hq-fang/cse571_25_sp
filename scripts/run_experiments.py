#!/usr/bin/env python3
import subprocess
import re
import csv

# the five noise factors we want to sweep
# default is 1
data_factors = [1/16, 1/4, 1, 4, 16]
filter_factors = [1/16, 1/4, 1, 4, 16]

# pattern to pull out the four stats at the end of each run
stat_re = {
    "mean_mpe":   re.compile(r"Mean MPE:\s*([0-9.]+)"),
    "std_mpe":    re.compile(r"Standard deviation of MPE:\s*([0-9.]+)"),
    "mean_anees": re.compile(r"Mean ANEES:\s*([0-9.]+)"),
    "std_anees":  re.compile(r"Standard deviation of ANEES:\s*([0-9.]+)"),
}

# where to dump results
output_csv = "exp_results.csv"

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "data_factor","filter_factor",
        "mean_mpe","std_mpe","mean_anees","std_anees",
    ])
    writer.writeheader()

    for df in data_factors:
        for ff in filter_factors:
            print(f"→ running data={df:g}, filter={ff:g} …", end="", flush=True)
            # call localization.py, no GUI, single run
            cmd = [
                "python", "localization.py", "pf",
                "--data-factor", str(df),
                "--filter-factor", str(ff),
                "--multi_run", "10",    # already returns single‐run stats
                "--seed", "0",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            out = proc.stdout + proc.stderr

            # parse out the four numbers
            stats = {"data_factor": df, "filter_factor": ff}
            for key, regex in stat_re.items():
                m = regex.search(out)
                if not m:
                    raise RuntimeError(f"Couldn’t find {key} in output!")
                stats[key] = float(m.group(1))

            writer.writerow(stats)
            print(" done.")

print(f"\nAll done! Results in {output_csv}")