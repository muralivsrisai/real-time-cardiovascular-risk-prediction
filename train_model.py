import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =======================
# Config
# =======================
DATA_PATH = "data/personal_health_data.csv"
MODEL_DIR = "models"
RISK_THRESHOLD = 60
TOP_K_FEATURES = 10
MANDATORY_FEATURES = ["Age"]


os.makedirs(MODEL_DIR, exist_ok=True)

# =======================
# Load Data
# =======================
df = pd.read_csv(DATA_PATH)

# =======================
# Encode categorical columns
# =======================
cat_cols = [
    "Gender",
    "Medical_Conditions",
    "Medication",
    "Smoker",
    "Alcohol_Consumption",
    "Mood"
]

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))

# =======================
# Create Risk Label
# =======================
df["Risk"] = np.where(df["Health_Score"] < RISK_THRESHOLD, 1, 0)

# =======================
# Automatic Feature Selection (with mandatory features)
# =======================
corr = df.corr(numeric_only=True)

excluded_cols = ["Health_Score", "Risk"]

selected_features = (
    corr["Health_Score"]
    .abs()
    .sort_values(ascending=False)
    .drop(labels=excluded_cols, errors="ignore")
    .head(TOP_K_FEATURES)
    .index
    .tolist()
)

# ✅ Force-include mandatory features (e.g., Age)
for feat in MANDATORY_FEATURES:
    if feat not in selected_features and feat in df.columns:
        selected_features.append(feat)

print("📌 Final model-selected features:")
for f in selected_features:
    print(" -", f)

# Save selected features (VERY IMPORTANT)
pd.DataFrame(selected_features, columns=["feature"]).to_csv(
    os.path.join(MODEL_DIR, "features.csv"),
    index=False
)

# =======================
# Prepare Data
# =======================
X = df[selected_features].values
y = df["Risk"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =======================
# Scaling
# =======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "wearable_scaler.pkl"))

# =======================
# Reshape for LSTM
# (samples, timesteps=1, features)
# =======================
X_train_3d = X_train_scaled.reshape(
    X_train_scaled.shape[0], 1, X_train_scaled.shape[1]
)
X_test_3d = X_test_scaled.reshape(
    X_test_scaled.shape[0], 1, X_test_scaled.shape[1]
)

# =======================
# Build Model
# =======================
model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(1, X_train_3d.shape[2])
    ),
    Conv1D(32, kernel_size=1, activation="relu"),
    MaxPooling1D(pool_size=1),
    Flatten(),

    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =======================
# Train
# =======================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

model.fit(
    X_train_3d,
    y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =======================
# Evaluate
# =======================
loss, acc = model.evaluate(X_test_3d, y_test, verbose=0)
print(f"✅ Test Accuracy: {acc:.4f}")

# =======================
# Save Model
# =======================
model.save(os.path.join(MODEL_DIR, "wearable_risk_model.keras"))

print("✅ Model, scaler, encoders, and feature list saved successfully.")
