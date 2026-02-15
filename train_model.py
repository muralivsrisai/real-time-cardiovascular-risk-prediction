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
TIME_STEPS = 10

os.makedirs(MODEL_DIR, exist_ok=True)

# =======================
# Load Data
# =======================
df = pd.read_csv(DATA_PATH)

# =======================
# Encode categorical columns
# =======================
cat_cols = [
    "Gender", "Medical_Conditions", "Medication",
    "Smoker", "Alcohol_Consumption", "Mood"
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
# Feature Selection
# =======================
corr = df.corr(numeric_only=True)
excluded = ["Health_Score", "Risk"]

selected_features = (
    corr["Health_Score"]
    .abs()
    .sort_values(ascending=False)
    .drop(labels=excluded, errors="ignore")
    .head(TOP_K_FEATURES)
    .index
    .tolist()
)

for f in MANDATORY_FEATURES:
    if f not in selected_features:
        selected_features.append(f)

pd.DataFrame(selected_features, columns=["feature"]).to_csv(
    os.path.join(MODEL_DIR, "features.csv"), index=False
)

print("📌 Selected Features:", selected_features)

# =======================
# Prepare Data
# =======================
X = df[selected_features].values
y = df["Risk"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# =======================
# Scaling
# =======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "wearable_scaler.pkl"))

# =======================
# Create Temporal Sequences
# =======================
def create_sequences(X, y, steps):
    Xs, ys = [], []
    for i in range(len(X) - steps):
        Xs.append(X[i:i+steps])
        ys.append(y[i+steps])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, TIME_STEPS)

# =======================
# Build Model
# =======================
model = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(TIME_STEPS, X_train_seq.shape[2])),
    Conv1D(32, kernel_size=1, activation="relu"),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# =======================
# Train
# =======================
early_stop = EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)

model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=40,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =======================
# Evaluate & Save
# =======================
loss, acc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"✅ Test Accuracy: {acc:.4f}")

model.save(os.path.join(MODEL_DIR, "wearable_risk_model.keras"))
print("✅ Training complete.")
