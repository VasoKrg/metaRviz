import pandas as pd, numpy as np, re

# ==== CONFIG ====
SRC = "C://Users//vaso0//Desktop//Meta_Analysis_Tables//MOCK_MAIN_TABLE.csv"   # your main table CSV
DEST = "C://Users//vaso0//Desktop//Meta_Analysis_Tables//Contingency_By_Classification.xlsx"

SETS = ["Training", "Testing", "Overall"]

# Classification schemes: which class is positive, which negative
CLASS_RULES = {
    "benign vs malignant": {"pos": ["malignant"], "neg": ["benign"]},
    "malignant vs normal":    {"pos": ["malignant"],    "neg": ["normal"]},
    "malignant vs non-malignant": {"pos": ["malignant"], "neg": ["benign", "normal"]},
}

# Expected class mix (proportions) per classification
MIX = {
    "malignant vs benign":           {"malignant": 0.5,  "benign": 0.5,  "normal": 0.0},
    "malignant vs normal":              {"malignant": 0.0,  "benign": 0.5,  "normal": 0.5},
    "malignant vs non-malignant":    {"malignant": 0.34, "benign": 0.33, "normal": 0.33},
}

# Performance bands
TARGETS = {
    "Training": {"sens": (0.85, 0.97), "spec": (0.85, 0.97), "auc_noise": 0.02},
    "Testing":  {"sens": (0.70, 0.92), "spec": (0.70, 0.92), "auc_noise": 0.03},
    "Overall":  {"auc_noise": 0.02},  # Overall metrics derived from confusion matrix
}

TOTAL_RANGE = (10, 100)   # total cases per (Study,Model,Set)

# ==== HELPERS ====
def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def pick_rule(classification: str):
    key = norm_key(classification)
    for pat, rule in CLASS_RULES.items():
        if pat in key:
            return pat, rule
    if "malignant" in key:
        return "malignant vs rest", {"pos": ["malignant"], "neg": ["benign", "normal"]}
    return "custom", {"pos": [key.split()[0]], "neg": []}

def split_models(s):
    if pd.isna(s): return []
    import re
    parts = re.split(r"[;,|]", str(s))
    return [p.strip() for p in parts if p.strip()]

def clamp01(x): return max(0.0, min(1.0, float(x)))

def sample_mix(label):
    base = MIX.get(label, {"malignant": 0.33, "benign": 0.34, "normal": 0.33})
    vals = {k: max(0.0, v + np.random.normal(0, 0.05)) for k, v in base.items()}
    s = sum(vals.values()) or 1.0
    return {k: v/s for k, v in vals.items()}

def draw_perf(set_name):
    lo1, hi1 = TARGETS[set_name]["sens"]
    lo2, hi2 = TARGETS[set_name]["spec"]
    sens = np.random.uniform(lo1, hi1)
    spec = np.random.uniform(lo2, hi2)
    return sens, spec

def derive_auc(sens, spec, noise):
    return clamp01((sens + spec)/2 + np.random.normal(0, noise))

def make_counts(total, mix):
    m = int(round(total * mix.get("malignant", 0)))
    b = int(round(total * mix.get("benign", 0)))
    n = int(round(total * mix.get("normal", 0)))
    drift = total - (m + b + n)
    if drift != 0:
        bucket = max([("malignant", m), ("benign", b), ("normal", n)], key=lambda t: t[1])[0]
        if bucket == "malignant": m += drift
        elif bucket == "benign":  b += drift
        else: n += drift
    return m, b, n

def confusion_from_perf(pos, neg, sens, spec):
    TP = int(round(sens * pos)); FN = pos - TP
    TN = int(round(spec * neg)); FP = neg - TN
    return max(0, TP), max(0, TN), max(0, FP), max(0, FN)

# ==== LOAD MAIN TABLE ====
df = pd.read_csv(SRC)
need = ["Study", "DOI", "Type of Study", "Classification", "Model"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"MainTable is missing columns: {missing}")

# ==== BUILD OUTPUT ====
out_by_class = {}

for _, row in df.iterrows():
    study, doi, tstudy = row["Study"], row["DOI"], row["Type of Study"]
    classification, model_cell = row["Classification"], row["Model"]
    models = split_models(model_cell) or ["Model"]
    label, rule = pick_rule(classification)
    mix = sample_mix(label)
    sheet_name = str(classification)[:31] if pd.notna(classification) else "Unclassified"
    rows = out_by_class.setdefault(sheet_name, [])

    for model in models:
        set_results = {}

        # Training + Testing
        for set_name in ["Training", "Testing"]:
            total = int(np.random.randint(*TOTAL_RANGE))
            malig, benign, normal = make_counts(total, mix)

            # enforce zero rules
            key = norm_key(classification)
            if "benign vs malignant" in key:
                normal = 0; total = malig + benign
            elif "malignant vs normal" or "normal vs malignant" in key:
                benign = 0; total = malig + normal
           

            # pos/neg pools
            pos_pool = sum({
                "malignant": malig, "benign": benign, "normal": normal
            }[k] for k in rule["pos"] if k in ["malignant","benign","normal"])
            neg_pool = total - pos_pool

            sens, spec = draw_perf(set_name)
            TP, TN, FP, FN = confusion_from_perf(pos_pool, neg_pool, sens, spec)
            acc = (TP + TN) / total if total else 0.0
            auc = derive_auc(sens, spec, TARGETS[set_name]["auc_noise"])

            set_results[set_name] = {
                "Study": study, "DOI": doi, "Type of Study": tstudy,
                "Set": set_name, "Total": total,
                "Malignant Breast Tissues": malig,
                "Benign Breast Tissues": benign,
                "Normal Breast Tissues": normal,
                "TP": TP, "TN": TN, "FP": FP, "FN": FN,
                "AUC": round(auc, 3),
                "Accuracy": round(acc, 3),
                "Sensitivity": round(sens, 3),
                "Specificity": round(spec, 3),
                "Model": model
            }

        # Overall = sum of Training + Testing
        tr, te = set_results["Training"], set_results["Testing"]
        total = tr["Total"] + te["Total"]
        malig = tr["Malignant Breast Tissues"] + te["Malignant Breast Tissues"]
        benign = tr["Benign Breast Tissues"] + te["Benign Breast Tissues"]
        normal = tr["Normal Breast Tissues"] + te["Normal Breast Tissues"]
        TP = tr["TP"] + te["TP"]; TN = tr["TN"] + te["TN"]
        FP = tr["FP"] + te["FP"]; FN = tr["FN"] + te["FN"]
        acc = (TP + TN) / total if total else 0.0
        sens = TP / (TP + FN) if (TP + FN) else 0.0
        spec = TN / (TN + FP) if (TN + FP) else 0.0
        auc = derive_auc(sens, spec, TARGETS["Overall"]["auc_noise"])

        set_results["Overall"] = {
            "Study": study, "DOI": doi, "Type of Study": tstudy,
            "Set": "Overall", "Total": total,
            "Malignant Breast Tissues": malig,
            "Benign Breast Tissues": benign,
            "Normal Breast Tissues": normal,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "AUC": round(auc, 3),
            "Accuracy": round(acc, 3),
            "Sensitivity": round(sens, 3),
            "Specificity": round(spec, 3),
            "Model": model
        }

        rows.extend(set_results.values())

# ==== WRITE EXCEL ====
with pd.ExcelWriter(DEST, engine="openpyxl") as w:
    for sheet, rows in out_by_class.items():
        df_sheet = pd.DataFrame(rows)
        cols = ["Study","DOI","Type of Study","Set","Total",
                "Malignant Breast Tissues","Benign Breast Tissues","Normal Breast Tissues",
                "TP","TN","FP","FN","AUC","Accuracy","Sensitivity","Specificity","Model"]
        df_sheet = df_sheet[cols]
        df_sheet.to_excel(w, sheet_name=sheet, index=False)

print(f"Created {DEST}")
