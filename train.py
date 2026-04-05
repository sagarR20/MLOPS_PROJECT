# ======================================
# 1. Import Libraries
# ======================================
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

# ======================================
# 2. Load Dataset
# ======================================
df = pd.read_csv("data/blood_cell_anomaly_detection.csv")

print("Original Shape:", df.shape)

# ======================================
# 3. Preprocessing
# ======================================
# Keep only numeric columns
df = df.select_dtypes(include=['float64', 'int64'])

# Fill missing values
df = df.fillna(df.mean())

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

print("Processed Shape:", data_scaled.shape)

# Train-test split
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

# ======================================
# 4. Build Autoencoder Model
# ======================================
input_dim = X_train.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),  # bottleneck
    layers.Dense(16, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

model.summary()

# ======================================
# 5. Train Model
# ======================================
history = model.fit(
    X_train, X_train,
    epochs=45,
    batch_size=32,
    validation_data=(X_test, X_test),
    shuffle=True
)

# ======================================
# 6. Reconstruction Error
# ======================================
reconstructions = model.predict(data_scaled)
mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)

# ======================================
# 7. Threshold Calculation
# ======================================
threshold = np.mean(mse) + 2 * np.std(mse)
print("Threshold:", threshold)

# Detect anomalies
anomalies = mse > threshold
print("Total anomalies detected:", np.sum(anomalies))

# ======================================
# 8. Save Model + Scaler + Threshold
# ======================================
model.save("model/autoencoder.h5")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(threshold, "model/threshold.pkl")

print("✅ Model, Scaler, Threshold saved successfully!")
