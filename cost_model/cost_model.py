# 🚀 TRAINING CODE: COST ESTIMATION MODEL

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_excel("vehicle_damage_dataset.xlsx")

# ==============================
# FIX BROKEN FORMAT (IMPORTANT)
# ==============================
if len(df.columns) == 1 or "Unnamed" in str(df.columns):
    col = df.columns[0]
    df = df[col].str.split(r'\s{2,}', expand=True)
    df.columns = [
        "Brand", "Model", "Part",
        "Part_Price", "Labor_Cost",
        "Paint_Cost", "Total_Estimated_Cost"
    ]

# ==============================
# CLEAN DATA
# ==============================
df["Part"] = df["Part"].astype(str).str.lower()
df["Model"] = df["Model"].astype(str).str.lower()

df["Part_Price"] = pd.to_numeric(df["Part_Price"], errors='coerce')
df["Labor_Cost"] = pd.to_numeric(df["Labor_Cost"], errors='coerce')
df["Paint_Cost"] = pd.to_numeric(df["Paint_Cost"], errors='coerce')
df["Total_Estimated_Cost"] = pd.to_numeric(df["Total_Estimated_Cost"], errors='coerce')

df.dropna(inplace=True)

# ==============================
# ADD SEVERITY
# ==============================
np.random.seed(42)
df["severity"] = np.random.uniform(0.3, 1.0, len(df))
df["adjusted_cost"] = df["Total_Estimated_Cost"] * df["severity"]

# ==============================
# ENCODE DATA
# ==============================
le_model = LabelEncoder()
le_part = LabelEncoder()

df["Model_encoded"] = le_model.fit_transform(df["Model"])
df["Part_encoded"] = le_part.fit_transform(df["Part"])

# ==============================
# TRAIN MODEL
# ==============================
X = df[["Model_encoded", "Part_encoded", "severity"]]
y = df["adjusted_cost"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("✅ Model trained successfully!")

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, "cost_model.pkl")
joblib.dump(le_model, "model_encoder.pkl")
joblib.dump(le_part, "part_encoder.pkl")

print("✅ Model saved successfully!")