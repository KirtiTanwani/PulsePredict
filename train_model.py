import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_save_model():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('cardio_train.csv', sep=';')

    # Feature Engineering
    # Calculate age in years
    df['age_years'] = (df['age'] / 365.25).astype(float)
    
    # Calculate BMI
    # height is in cm and weight is in kg
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Define features and target
    # CRITICAL: This exact order must be maintained in the frontend
    features = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
    target = 'cardio'

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    print("Training Logistic Regression Model...")
    # Increase max_iter to ensure convergence
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Save Model and Scaler
    print("Saving model and scaler to cardio_model.pkl...")
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features
    }
    joblib.dump(model_data, 'cardio_model.pkl')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
