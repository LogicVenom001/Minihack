import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("SMART METER POWER CONSUMPTION PROFILING")

# Load dataset
df = pd.read_csv("household_power_consumption.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nShape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# Convert columns to numeric
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

# Remove missing values
df.dropna(inplace=True)

print("\nData Shape After Cleaning")
print(df.shape)

# Summary statistics
print("\nStatistics")
print(df[numeric_cols].describe())

# Histograms
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=20)
    plt.title(col)
    plt.savefig(f"{col}_histogram.png")
    plt.close()

# Correlation matrix
corr = df[numeric_cols].corr()

print("\nCorrelation Matrix")
print(corr)

# Heatmap
plt.figure(figsize=(8,6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# Boxplot
plt.figure(figsize=(7,5))
plt.boxplot(df["Global_active_power"])
plt.title("Power Consumption Outliers")
plt.savefig("boxplot_power.png")
plt.show()

# Power trend
plt.figure(figsize=(10,5))
plt.plot(df["Global_active_power"].head(1000))
plt.title("Global Active Power Trend")
plt.xlabel("Records")
plt.ylabel("Power")
plt.savefig("power_trend.png")
plt.show()

# Machine Learning
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Actual vs Predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.savefig("actual_vs_predicted.png")
plt.show()

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(8,5))
plt.bar(importance["Feature"], importance["Coefficient"])
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nProject Completed Successfully")

print("\nKey Insights:")
print("- Missing values were handled.")
print("- Power consumption trends were analyzed.")
print("- Global Intensity showed the strongest relation with power consumption.")
print("- Outliers were identified.")
print("- A Linear Regression model was built for prediction.")
