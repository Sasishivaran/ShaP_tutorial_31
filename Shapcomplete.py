#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Cell 1: generate synthetic churn dataset, train a pipeline, save model and CSV
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

RANDOM_SEED = 42

# 1) Create synthetic dataset
np.random.seed(RANDOM_SEED)
n = 1000
df = pd.DataFrame({
    "customer_id": [f"CUST_{i:05d}" for i in range(n)],
    "age": np.random.randint(18, 80, size=n),
    "tenure_months": np.random.randint(0, 72, size=n),
    "monthly_charges": np.round(np.random.uniform(20, 150, size=n), 2),
    "num_support_tickets": np.random.poisson(1.2, size=n),
    "contract_type": np.random.choice(["month-to-month", "one-year", "two-year"], size=n, p=[0.6,0.25,0.15]),
    "payment_method": np.random.choice(["electronic_check","mailed_check","bank_transfer","credit_card"], size=n),
    "has_internet": np.random.choice([0,1], size=n, p=[0.1,0.9])
})

# 2) Create target 'churn'
logit = (
    0.02 * (df["monthly_charges"] - df["monthly_charges"].mean()) +
    -0.03 * (df["tenure_months"]) +
    0.5 * df["num_support_tickets"] +
    0.6 * (df["contract_type"] == "month-to-month").astype(int) +
    0.3 * (df["has_internet"] == 0).astype(int)
)
prob = 1 / (1 + np.exp(- ( -1.0 + logit / 10 )))
df["churn"] = (np.random.rand(n) < prob).astype(int)

# 3) Shuffle rows and save CSV
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
df.to_csv("customer_data.csv", index=False)
print("Saved customer_data.csv with shape:", df.shape)

# 4) Prepare features and pipeline
feature_cols = ["age", "tenure_months", "monthly_charges", "num_support_tickets", "contract_type", "payment_method", "has_internet"]
X = df[feature_cols]
y = df["churn"]

numeric_features = ["age", "tenure_months", "monthly_charges", "num_support_tickets", "has_internet"]
categorical_features = ["contract_type", "payment_method"]

numeric_transformer = StandardScaler()
# Use sparse_output=False for newer scikit-learn; if your sklearn is older, you can use sparse=False
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)

pipeline = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", clf)
])

# 5) Train pipeline
pipeline.fit(X, y)
print("Trained pipeline.")

# 6) Show feature names after preprocessing (robust across sklearn versions)
try:
    feat_names = pipeline.named_steps["prep"].get_feature_names_out()
except Exception:
    # Fallback: build feature names manually
    cat_names = pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
    feat_names = list(numeric_features) + list(cat_names)
print("Number of features after preprocessing:", len(feat_names))
print(feat_names[:20])

# 7) Save model
joblib.dump(pipeline, "churn_model.pkl")
print("Saved churn_model.pkl")


# In[15]:


# Cell 1: load model and data
import joblib
import pandas as pd
import numpy as np

model = joblib.load("churn_model.pkl")
data = pd.read_csv("customer_data.csv")

print("Model type:", type(model))
print("Data shape:", data.shape)


# In[16]:


# Cell 2: prepare X/y, split, predict
from sklearn.model_selection import train_test_split

feature_cols = ["age", "tenure_months", "monthly_charges",
                "num_support_tickets", "contract_type", "payment_method", "has_internet"]

X = data[feature_cols]
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

y_pred = model.predict(X_test)
print("X_test shape:", X_test.shape)


# In[17]:


# Cell 3: find misclassified rows; fallback to sample if none
results = X_test.copy()
results["true"] = y_test.values
results["pred"] = y_pred
misclassified = results[results["true"] != results["pred"]].reset_index(drop=True)

if misclassified.shape[0] == 0:
    # fallback: sample up to 10 rows for demonstration
    print("No misclassifications found — using sample of test rows for SHAP demo.")
    misclassified = results.sample(n=min(10, len(results)), random_state=42).reset_index(drop=True)
    fallback_used = True
else:
    fallback_used = False

X_mis_raw = misclassified.drop(columns=["true", "pred"])
print("Misclassified (or sampled) rows:", X_mis_raw.shape[0])


# In[20]:


# Cell 4: transform to numeric arrays and compute SHAP using TreeExplainer on the tree estimator
import shap
import numpy as np

# Ensure expected feature order
feature_cols = ["age", "tenure_months", "monthly_charges",
                "num_support_tickets", "contract_type", "payment_method", "has_internet"]

X_train = X_train.loc[:, feature_cols]
X_mis_raw = X_mis_raw.loc[:, feature_cols]

# Get pipeline pieces
prep = model.named_steps.get("prep", None)
clf = model.named_steps.get("clf", None)
if prep is None or clf is None:
    raise RuntimeError("Expected pipeline with steps named 'prep' and 'clf'.")

# Transform to numeric arrays (OneHotEncoder output may be sparse -> convert)
X_train_trans = prep.transform(X_train)
X_mis_trans = prep.transform(X_mis_raw)

if hasattr(X_train_trans, "toarray"):
    X_train_trans = X_train_trans.toarray()
if hasattr(X_mis_trans, "toarray"):
    X_mis_trans = X_mis_trans.toarray()

print("X_train_trans.shape:", X_train_trans.shape)
print("X_mis_trans.shape:", X_mis_trans.shape)

# Use TreeExplainer on the trained tree estimator (RandomForest)
explainer = shap.TreeExplainer(clf)
# For multiclass RandomForest, shap_values will be a list; for binary it may return array
sv = explainer.shap_values(X_mis_trans)

# Normalize to shap_matrix (n_samples, n_transformed_features)
if isinstance(sv, list):
    # pick class 1 (positive) contributions if present
    class_idx = 1 if len(sv) > 1 else 0
    shap_matrix = np.asarray(sv[class_idx])
else:
    shap_matrix = np.asarray(sv)

print("shap_matrix.shape:", shap_matrix.shape)
n_trans_features = X_train_trans.shape[1]
if shap_matrix.shape[1] != n_trans_features:
    raise ValueError(f"SHAP feature dim {shap_matrix.shape[1]} != expected {n_trans_features}")

# Derive transformed feature names for plotting
try:
    trans_feature_names = prep.get_feature_names_out()
except Exception:
    numeric_features = ["age","tenure_months","monthly_charges","num_support_tickets","has_internet"]
    cat_features = ["contract_type","payment_method"]
    cat_names = prep.named_transformers_["cat"].get_feature_names_out(cat_features)
    trans_feature_names = list(numeric_features) + list(cat_names)

print("Number of transformed features:", len(trans_feature_names))


# In[22]:


# Cell 5 (fixed): waterfall for one example — pick class index then plot single explanation
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

row_idx = 0        # change to inspect other examples
class_idx = 1      # for binary positive class; set 0/1 as needed for your task

# shap_matrix shape should be (n_samples, n_trans_features)
# X_mis_trans is the transformed numeric array (n_samples, n_trans_features)
# trans_feature_names is the list of transformed feature names
# explainer is the TreeExplainer instance used earlier

# Get the 1-D SHAP values for this row (ensure shape is (n_features,))
row_vals = np.asarray(shap_matrix[row_idx])
if row_vals.ndim == 2:
    # if shape is (n_classes, n_features) or (n_features, n_classes), try to pick correct orientation
    # common case: sv produced list earlier, but defensively handle both
    if row_vals.shape[0] == len(trans_feature_names) and row_vals.shape[1] <= 5:
        # (n_features, n_classes) -> take column for class_idx
        row_vals = row_vals[:, class_idx]
    elif row_vals.shape[1] == len(trans_feature_names) and row_vals.shape[0] <= 5:
        # (n_classes, n_features) -> take row for class_idx
        row_vals = row_vals[class_idx]
    else:
        # fallback: flatten to 1-D (not ideal); prefer explicit shapes above
        row_vals = row_vals.ravel()[: len(trans_feature_names)]

# Determine scalar base value for the chosen class
base_val = None
try:
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)):
        # pick class index if array-like
        base_val = float(ev[class_idx]) if len(ev) > 1 else float(ev[0])
    else:
        base_val = float(ev)
except Exception:
    # fallback: compute model prediction mean for that class
    probs = clf.predict_proba(X_train_trans)[:, class_idx]
    base_val = float(probs.mean())

# Build Explanation object and plot waterfall
expl = shap.Explanation(values=row_vals,
                        base_values=base_val,
                        data=X_mis_trans[row_idx],
                        feature_names=trans_feature_names)

shap.plots.waterfall(expl)


# In[26]:


# Align SHAP array robustly and plot summary (handles shapes like (n,classes,features) or (n,features,classes))
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Preconditions: sv (TreeExplainer output), X_mis_trans (n_samples, n_trans_features), trans_feature_names
print("X_mis_trans.shape:", getattr(X_mis_trans, "shape", None))
print("sv type:", type(sv))
arr = np.asarray(sv)
print("sv raw shape:", arr.shape)

# Normalize to (n_samples, n_features)
if arr.ndim == 3:
    n, a, b = arr.shape
    if a == X_mis_trans.shape[1] and b != X_mis_trans.shape[1]:
        # arr is (n, n_features, n_classes) -> move classes to middle: (n, n_classes, n_features)
        arr = arr.transpose(0, 2, 1)
        print("Transposed sv to shape:", arr.shape)
    # Now arr should be (n, n_classes, n_features)
    if arr.shape[1] > 1:
        class_idx = 1 if arr.shape[1] > 1 else 0
        sm = arr[:, class_idx, :]
    else:
        sm = arr[:, 0, :]
elif arr.ndim == 2:
    # arr is (n_samples, n_features) already
    sm = arr
else:
    raise ValueError(f"Unexpected sv ndim={arr.ndim} shape={arr.shape}")

print("Aligned SHAP (sm) shape:", sm.shape, "expected features:", X_mis_trans.shape[1])

# Final check
if not (sm.ndim == 2 and sm.shape[1] == X_mis_trans.shape[1]):
    raise ValueError(f"After alignment, SHAP shape is {sm.shape} but expected (n_samples, n_features) = {X_mis_trans.shape}.")

# Plot summary (use transformed numeric inputs for coloring)
try:
    shap.summary_plot(sm, X_mis_trans, feature_names=trans_feature_names, show=True)
except Exception as e:
    mean_abs = pd.Series(np.abs(sm).mean(axis=0), index=trans_feature_names).sort_values(ascending=True)
    mean_abs.tail(30).plot(kind="barh", figsize=(9, max(4, len(mean_abs)*0.2)))
    plt.title("Mean absolute SHAP (fallback)")
    plt.xlabel("mean |SHAP value|")
    plt.show()
    print("Fallback used due to:", e)


# In[27]:


# quick checks
print(len(trans_feature_names), "transformed features")
from pandas import DataFrame
corr = DataFrame(X_mis_trans, columns=trans_feature_names).corr().abs()
high_corr = [(i,j,corr.loc[i,j]) for i in corr.columns for j in corr.columns if i!=j and corr.loc[i,j]>0.85]
high_corr[:10]


# In[29]:


for i, n in enumerate(trans_feature_names[:200]):
    print(i, n)
# then pick the exact name printed and use it below


# In[30]:


# auto-find a transformed feature containing "monthly" (case-insensitive)
match = [n for n in trans_feature_names if "monthly" in n.lower()]
print("matches:", match)
shap.dependence_plot(match[0], sm, X_mis_trans, feature_names=trans_feature_names)


# In[31]:


# replace IDX with the integer index you saw from the name list
IDX = 0
shap.dependence_plot(IDX, sm, X_mis_trans, feature_names=trans_feature_names)


# In[ ]:




