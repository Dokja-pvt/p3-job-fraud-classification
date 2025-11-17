# Libraries
import pandas as pd # Data handling library
import joblib # Tool for loading our saved model.
import matplotlib.pyplot as plt # Tool for plotting Confusion Matrix.
from sklearn.model_selection import train_test_split # Tool to split data for testing.
from data_processing import clean_and_combine_text # function that cleans text and combines text columns.
from sklearn.metrics import classification_report, ConfusionMatrixDisplay # function used to grade model performance and visualize results.

# Execution Block
if __name__ == "__main__":

    # Loading data from relative path
    df = pd.read_csv('../data/fake_job_postings.csv')

    # Manual Text Cleaning & Combining logic from train.py:
    text_cols = ['title', 'description', 'requirements']
    df[text_cols] = df[text_cols].fillna('') # Fills missing text (NaNs) with a blank string ('').
    df['combined_text'] = df['title'] + ' ' + df['description'] + ' ' + df['requirements'] # Creates the single 'combined_text' feature for the TfidfVectorizer.

    # 3. Create X (features) and y (target).
    X = df.drop(['fraudulent', 'title', 'description', 'requirements'], axis=1)
    y = df['fraudulent']

    # 4. Split the data. This must be IDENTICAL to the split used in train.py.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # CRUCIAL for preserving the 4.8% imbalance
    )

    print("Evaluation data loaded and split successfully.")
    
    print("Loading 'race car' (best_pipeline.pkl)...")

    # Define the relative path to the final model saved during Phase 3.
    model_path = r'D:\College\Major Project\p3-job-fraud-classification\results\best_pipeline.pkl'
    # Use joblib to load the model file from the path.
    pipeline = joblib.load(model_path)

    print("Best pipeline successfully loaded.")