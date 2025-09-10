# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("Cost_of_Living_Index_2022.csv")

# Target column
target_col = "Cost of Living Index"

# Features (drop Rank + leakage columns)
drop_cols = ["Rank", "Cost of Living Plus Rent Index", target_col]
features = [c for c in df.select_dtypes(include=[float, int]).columns if c not in drop_cols]

X = df[features]
y = df[target_col]

# -------------------------------
# 2. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 3. Train Model
# -------------------------------
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 4. Evaluate
# -------------------------------
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)
print(f"✅ Model trained. RMSE={rmse:.2f}, R²={r2:.2f}")

# -------------------------------
# 5. Save Model
# -------------------------------
joblib.dump({"model": model, "features": features, "target": target_col},
            "models/cost_of_living_model.pkl")
print("📦 Model saved to models/cost_of_living_model.pkl")

import json

# Save metrics to file
metrics = {"RMSE": rmse, "R2": r2}
with open("models/training_metrics.json", "w") as f:
    json.dump(metrics, f)

print("📊 Training metrics saved to models/training_metrics.json")

