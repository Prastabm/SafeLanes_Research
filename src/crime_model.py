import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load data
df = pd.read_csv("data/cleaned_crime_data2.csv", parse_dates=['datetime'])

# Feature engineering
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month

# Drop rows with missing target or features
df.dropna(subset=['crime_category', 'description'], inplace=True)

# Features and target
X = df[['Latitude', 'Longitude', 'hour', 'day_of_week', 'month', 'city', 'description']]
y = df['crime_category']

# Define preprocessing for categorical data
categorical_features = ['city', 'description']
numeric_features = ['Latitude', 'Longitude', 'hour', 'day_of_week', 'month']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # keep numeric features
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print("🔍 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Optional: save the model
import joblib
joblib.dump(pipeline, 'crime_category_model.pkl')
print("✅ Model saved as crime_category_model.pkl")
