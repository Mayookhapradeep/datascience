import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# 1) Load dataset
df = pd.read_csv("iris.csv")

section("1) First 10 Rows")
print(df.head(10))

section("2) Dataset Info")
df.info()

section("3) Summary Statistics")
print(df.describe())

# 4) Check missing values
section("4) Missing Values Per Column")
print(df.isnull().sum())

# Handle missing values
df = df.dropna()

# 5) Encode target labels
section("5) Categorical Encoding")
encoder = LabelEncoder()
df["species_encoded"] = encoder.fit_transform(df["species"])

print(df[["species", "species_encoded"]].head())

# 6) Split into features and target
section("6) Split Features and Target")
X = df.drop(columns=["species", "species_encoded"])
y = df["species_encoded"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# 7) Train-test split
section("7) Train-Test Split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8) Feature scaling
section("8) Feature Scaling")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9) Build neural network
section("9) Build Model")
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")
])

model.summary()

# 10) Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 11) Train model
section("10) Training")
model.fit(X_train, y_train, epochs=30, batch_size=16)

# 12) Evaluate model
section("11) Evaluation")
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# 13) Predictions
section("12) Predictions")
y_pred = np.argmax(model.predict(X_test), axis=1)

print("Predicted:", y_pred[:10])
print("Actual:   ", y_test.values[:10])