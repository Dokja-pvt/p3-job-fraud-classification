#libraries
import pandas as pd # Data handling library
import joblib # Used to save and load the final trained model object.
from sklearn.model_selection import train_test_split # splits data into training and testing sets.
from sklearn.pipeline import Pipeline # Organizes multiple sequential processing steps into one object.
from sklearn.model_selection import GridSearchCV # Tool for finding the best model parameters (hyperparameter tuning).
from sklearn.linear_model import LogisticRegression # A classification algorithm (our baseline model).
from sklearn.ensemble import RandomForestClassifier # ensemble classification algorithm (our advanced model).

# Phase 2 Preprocessing Engine
from data_processing import create_preprocessor # Imports the function that returns our fully built ColumnTransformer engine.

# Execution Block
if __name__ == "__main__":
    # Loading data from relative path
    df = pd.read_csv('../data/fake_job_postings.csv')

    # Text Cleaning & Combining
    # Fill missing values in the three text columns with empty string
    text_cols = ['title', 'description', 'requirements']
    df[text_cols] = df[text_cols].fillna('')

    # Create the new combined_text column by joining the three text columns
    df['combined_text'] = df['title'] + ' ' + df['description'] + ' ' + df['requirements']

    # Separate X and Y
    # Define X, dropping the target and the now redundant individual text columns.
    X = df.drop('fraudulent', axis=1)
    # Define the target variable (y).
    y = df['fraudulent']
    
    # Data Split
    # to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,       # Uses 20% for testing.
        random_state=42,     # Ensures reproducible results.
        stratify=y           # CRUCIAL: Preserves the 4.8% imbalance in both sets.
    )

    print("Test data loaded, cleaned, and split successfully.")
    
    # "Install" the Phase 2 engine
    preprocessor = create_preprocessor()
    
    print("Preprocessor engine successfully imported and created.")