import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 60)
print("SMART METER POWER CONSUMPTION PROFILING")
print("=" * 60)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("household_power_consumption.csv")

# ---------------------------------------------------
# BASIC EXPLORATION
# ---------------------------------------------------

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nMISSING VALUES")
print(df.isnull().sum())

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

numeric_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(inplace=True)

print("\nDATASET SHAPE AFTER CLEANING")
print(df.shape)

# ---------------------------------------------------
# DESCRIPTIVE STATISTICS
# ---------------------------------------------------

print("\nSTATISTICAL SUMMARY")
print(df[numeric_cols].describe())

# ---------------------------------------------------
# HISTOGRAMS
# ---------------------------------------------------

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=20)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{col}_histogram.png")
    plt.close()

print("\nHistogram images saved successfully.")

# ---------------------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------------------

corr = df[numeric_cols].corr()

print("\nCORRELATION MATRIX")
print(corr)

# ---------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------

plt.figure(figsize=(8,6))

plt.imshow(corr, cmap="coolwarm")

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()

# ---------------------------------------------------
# OUTLIER DETECTION
# ---------------------------------------------------

plt.figure(figsize=(8,5))

plt.boxplot(df["Global_active_power"])

plt.title("Outlier Detection - Global Active Power")

plt.ylabel("Power Consumption")

plt.savefig("boxplot_power.png")

plt.show()

# ---------------------------------------------------
# POWER TREND
# ---------------------------------------------------

plt.figure(figsize=(10,5))

plt.plot(df["Global_active_power"].head(1000))

plt.title("Global Active Power Trend")

plt.xlabel("Records")

plt.ylabel("Power Consumption")

plt.savefig("power_trend.png")

plt.show()

# ---------------------------------------------------
# MACHINE LEARNING MODEL
# ---------------------------------------------------

X = df[
    [
        "Voltage",
        "Global_reactive_power",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]
]

y = df["Global_active_power"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")

print("Mean Absolute Error (MAE):", mae)

print("R2 Score:", r2)

# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Power")

plt.ylabel("Predicted Power")

plt.title("Actual vs Predicted")

plt.tight_layout()

plt.savefig("actual_vs_predicted.png")

plt.show()

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFEATURE IMPORTANCE")

print(importance)

# ---------------------------------------------------
# FEATURE IMPORTANCE GRAPH
# ---------------------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Coefficient"]
)

plt.title("Feature Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

# ---------------------------------------------------
# INSIGHTS
# ---------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nKEY INSIGHTS")

print("1. Dataset cleaned successfully.")

print("2. Missing values removed.")

print("3. Power consumption distribution analyzed.")

print("4. Correlation between electrical parameters studied.")

print("5. Global Intensity has very strong relationship with Global Active Power.")

print("6. Outliers identified using boxplot.")

print("7. Machine Learning model trained successfully.")

print("8. Power consumption predictions generated.")

print("9. Feature importance analyzed.")

print("10. Visual insights generated for decision making.")
