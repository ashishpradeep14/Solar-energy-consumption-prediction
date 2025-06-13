

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import json

def create_dataset():
    """Generates the synthetic dataset."""
    np.random.seed(42)
    num_samples = 500
    data = {
        'Developer': np.random.choice(['SunPower', 'FirstSolar', 'CanadianSolar', 'TrinaSolar'], size=num_samples),
        'Region': np.random.choice(['Southwest', 'Southeast', 'Midwest', 'Northeast'], size=num_samples),
        'Equipment_Type': np.random.choice(['Monocrystalline', 'Polycrystalline', 'Thin-Film'], size=num_samples),
        'System_Size_MW': np.random.uniform(1, 100, size=num_samples),
        'Solar_Irradiance_kWh_m2_year': np.random.uniform(1400, 2200, size=num_samples),
        'Panel_Efficiency_pct': np.random.uniform(17, 23, size=num_samples),
        'Shading_Factor': np.random.uniform(0.95, 1.0, size=num_samples),
        'Average_Temperature_C': np.random.uniform(10, 30, size=num_samples)
    }
    df = pd.DataFrame(data)
    base_production = (df['System_Size_MW'] * 
                       df['Solar_Irradiance_kWh_m2_year'] * 
                       (df['Panel_Efficiency_pct'] / 100) *
                       df['Shading_Factor'] *
                       0.15)
    noise = np.random.normal(0, 200, size=num_samples)
    df['Annual_Energy_Production_GWh'] = (base_production + noise).clip(lower=0)
    return df

def main():
    print("Creating dataset...")
    df = create_dataset()
    
    # --- Define Features and Target ---
    X = df.drop('Annual_Energy_Production_GWh', axis=1)
    y = df['Annual_Energy_Production_GWh']
    
    # --- Preprocessing Pipeline ---
    numerical_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
        
    # --- Model Pipeline ---
    # Using Gradient Boosting as it was the best model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', GradientBoostingRegressor(random_state=42))])
    
    # --- Train the model on the ENTIRE dataset ---
    # For deployment, we train on all available data to make the model as robust as possible.
    print("Training model on the full dataset...")
    model_pipeline.fit(X, y)
    print("Model training complete.")

    # --- Save the Model ---
    model_filename = 'model/solar_production_model.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"Model saved to {model_filename}")

    # --- Save Metadata for the UI ---
    # This is crucial for populating dropdowns and ensuring column order.
    metadata = {
        "columns": X.columns.tolist(),
        "categorical_options": {
            col: df[col].unique().tolist() for col in categorical_cols
        },
        "numerical_ranges": {
            col: [df[col].min(), df[col].max()] for col in numerical_cols
        }
    }
    metadata_filename = 'model/model_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_filename}")

    print("\nTraining and saving process finished successfully!")

if __name__ == '__main__':
    main()