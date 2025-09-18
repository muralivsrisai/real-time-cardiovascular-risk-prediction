# train_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# =======================
# Config
# =======================
RISK_THRESHOLD = 60  # configurable threshold for defining "risky"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =======================
# Load Data
# =======================
df = pd.read_csv("data/personal_health_data.csv")

# =======================
# Encode categorical features
# =======================
cat_cols = ["Gender", "Medical_Conditions", "Medication", "Smoker", "Alcohol_Consumption", "Mood"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders
joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))

# =======================
# Create Target Variable
# =======================
df["Risk"] = np.where(df["Health_Score"] < RISK_THRESHOLD, 1, 0)

# =======================
# Feature Selection (Top 10 correlated with Health_Score)
# =======================
corr = df.corr(numeric_only=True)
top_features = (
    corr["Health_Score"]
    .abs()
    .sort_values(ascending=False)
    .drop("Health_Score")
    .head(10)
    .index
    .tolist()
)

print("📌 Selected features:", top_features)

X = df[top_features].values
y = df["Risk"].values

# =======================
# Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# Scaling
# =======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "wearable_scaler.pkl"))

# Reshape to 3D for LSTM
X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# =======================
# Build Model
# =======================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train_3d.shape[2])),
    Conv1D(32, kernel_size=1, activation='relu'),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# =======================
# Train Model
# =======================
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(
    X_train_3d, y_train,
    validation_data=(X_test_3d, y_test),
    epochs=40,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# =======================
# Evaluate
# =======================
loss, acc = model.evaluate(X_test_3d, y_test, verbose=0)
print(f"✅ Test accuracy: {acc:.4f}")

# =======================
# Save Model & Artifacts
# =======================
# Save in TensorFlow SavedModel format (no .h5 issues)
model.save(os.path.join(MODEL_DIR, "wearable_risk_model"))

# Save selected features
pd.DataFrame(top_features, columns=["feature"]).to_csv(
    os.path.join(MODEL_DIR, "features.csv"), index=False
)

print("✅ Model, scaler, encoders, and feature list saved successfully.")
