import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load local CSV
df = pd.read_csv("auto-mpg.csv")

# Handle missing values
df.dropna(inplace=True)



# If horsepower is object → convert
if df['horsepower'].dtype == 'object':
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df.dropna(inplace=True)

df.drop(columns=['car name'], inplace=True)

# Split
X = df.drop(columns=['mpg'])
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))

# Save model
joblib.dump(pipe, "mpg_pipeline.pkl")