import pandas as pd
from pathlib import Path

# --- Paths ---
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_PATH = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_PATH.mkdir(parents=True, exist_ok=True)

# --- Load raw data ---
train = pd.read_csv(RAW_PATH / "data_set_ALL_AML_train.csv", header=None)
independent = pd.read_csv(RAW_PATH / "data_set_ALL_AML_independent.csv", header=None)
actual = pd.read_csv(RAW_PATH / "actual.csv")

def process_file(df):
    """Remove 'call' columns and extract numeric sample IDs."""
    cols = df.iloc[0, 1:].tolist()
    valid_idx = [i for i, c in enumerate(cols) if c != "call"]
    samples = [cols[i] for i in valid_idx]
    genes = df.iloc[1:, 0].tolist()
    data = df.iloc[1:, [i + 1 for i in valid_idx]].T
    data.columns = genes
    data.index = samples
    return data

# --- Process train and independent sets ---
train_data = process_file(train)
ind_data = process_file(independent)

# --- Combine them ---
combined = pd.concat([train_data, ind_data])
print("Combined shape before trimming:", combined.shape)

# --- Keep only labeled patients from actual.csv ---
labeled_samples = set(actual["patient"].astype(str))
combined = combined.loc[combined.index.astype(str).isin(labeled_samples)]

# --- Add numeric labels ---
label_map = actual.set_index(actual["patient"].astype(str))["cancer"].map({"AML": 0, "ALL": 1})
combined["label"] = combined.index.astype(str).map(label_map)

print("After labeling:", combined["label"].value_counts())

# --- Save full dataset ---
out_full = PROC_PATH / "golub_combined.csv"
combined.to_csv(out_full)
print(f" Saved full dataset to {out_full}")



# --- Balance dataset (25 AML + 25 ALL) ---
aml_samples = combined[combined["label"] == 0]
all_samples = combined[combined["label"] == 1]

if len(aml_samples) >= 25 and len(all_samples) >= 25:
    aml_bal = aml_samples.sample(n=25, random_state=42)
    all_bal = all_samples.sample(n=25, random_state=42)
    balanced = pd.concat([aml_bal, all_bal]).sample(frac=1, random_state=42)

    out_bal = PROC_PATH / "golub_balanced.csv"
    balanced.to_csv(out_bal)
    print(f" Saved balanced dataset to {out_bal}")
    print("Balanced counts:\n", balanced["label"].value_counts())
else:
    print(" Not enough samples to create balanced dataset (need ≥25 per class).")

print("Final shape:", combined.shape)

