import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load data

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

columns = ['mpg','cylinders','displacement','horsepower','weight',
           'acceleration','model_year','origin','car_name']

df = pd.read_csv(url, names=columns, sep=r'\s+', na_values='?')

df.dropna(inplace=True)
df['horsepower'] = df['horsepower'].astype(float)
df.drop(columns=['car_name'], inplace=True)

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

# -------------------------
# Evaluate
# -------------------------
y_pred = pipe.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))

# -------------------------
# Save ONE file
# -------------------------
joblib.dump(pipe, "mpg_pipeline.pkl")
