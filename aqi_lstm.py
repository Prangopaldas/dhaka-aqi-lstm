import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
base_file = 'updated_data_preprocecing_24.csv'
filtered_file = 'dhaka_division_filtered_data.csv'
target_col = 'AQI_Bucket'

# Re-create filtered file if needed
if os.path.exists(filtered_file):
    os.remove(filtered_file)

if not os.path.exists(filtered_file):
    df = pd.read_csv(base_file)
    keywords = ['Dhaka', 'Narsingdi', 'Gazipur', 'Narayanganj', 'Faridpur', 'Gopalganj',
                'Kishoreganj', 'Madaripur', 'Manikganj', 'Munshiganj', 'Rajbari',
                'Shariatpur', 'Tangail']
    df = df[df['City'].str.contains('|'.join(keywords), case=False, na=False)]
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df.to_csv(filtered_file, index=False)

df = pd.read_csv(filtered_file)
print("Columns in dataset:", df.columns.tolist())
print("\nOriginal distribution:")
print(df[target_col].value_counts())

# Feature engineering
df['PM2.5_PM10_Ratio'] = df['PM2.5'] / (df['PM10'] + 1e-5)
df['Gas_Pollution_Index'] = df['SO2'] * df['NO2'] * df['CO']
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
df['Is_Peak_Traffic'] = df['Hour'].apply(lambda h: int((8 <= h <= 11) or (17 <= h <= 20)))

# Step 1: Drop AQI along with metadata to prevent target leakage
drop_cols = ['Date', 'Time', 'Datetime', 'Warning', 'Alert', 'Year', 'AQI']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Split: train / val / test
train_full, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
train_df, val_df = train_test_split(train_full, test_size=0.1, random_state=42, stratify=train_full[target_col])

# Imputation
num_cols = train_df.select_dtypes(include=[np.number]).columns
medians = train_df[num_cols].median()
for d in (train_df, val_df, test_df):
    d[num_cols] = d[num_cols].fillna(medians)

# Oversample train set
max_size = train_df[target_col].value_counts().max()
parts = []
for cls, grp in train_df.groupby(target_col):
    if len(grp) < max_size:
        grp = grp.sample(n=max_size, replace=True, random_state=42)
    parts.append(grp)
train_df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

# Encoding
city_map = {c: i for i, c in enumerate(sorted(train_df['City'].astype(str).unique()))}
for d in (train_df, val_df, test_df):
    d['City_Encoded'] = d['City'].astype(str).map(city_map).fillna(-1)

target_encoder = LabelEncoder().fit(df[target_col].astype(str))
class_names = target_encoder.classes_
num_classes = len(class_names)

def make_xy(d):
    X = d.drop(columns=[target_col, 'City'], errors='ignore')
    y = target_encoder.transform(d[target_col].astype(str))
    return X, y

X_train, y_train = make_xy(train_df)
X_val, y_val = make_xy(val_df)
X_test, y_test = make_xy(test_df)
feature_cols = X_train.columns
X_val, X_test = X_val[feature_cols], X_test[feature_cols]

# Scaling
scaler = StandardScaler().fit(X_train)
def prep(X):
    Xs = scaler.transform(X)
    return Xs.reshape(Xs.shape[0], 1, Xs.shape[1])

X_train, X_val, X_test = prep(X_train), prep(X_val), prep(X_test)
y_train_c = to_categorical(y_train, num_classes)
y_val_c = to_categorical(y_val, num_classes)

# Build and Train LSTM
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, activation='tanh'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train_c, validation_data=(X_val, y_val_c),
                    epochs=30, batch_size=64, callbacks=[early_stop], verbose=1)

# Evaluation
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
acc = accuracy_score(y_test, y_pred)
print("\n==================== TEST RESULTS ====================")
print(f"Test Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
