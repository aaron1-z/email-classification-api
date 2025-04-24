# models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os
from utils import mask_pii # Import the masking function

# --- !!! IMPORTANT: SET YOUR COLUMN NAMES HERE !!! ---
# Column names derived from the CSV snippet provided by the user
ACTUAL_EMAIL_COLUMN_NAME = "email"  # Column name for email content found in CSV
ACTUAL_CATEGORY_COLUMN_NAME = "type" # Column name for email category found in CSV
# --- !!! IMPORTANT: END OF COLUMN NAME SETTINGS !!! ---


DATA_PATH = "combined_emails_with_natural_pii.csv"
MODEL_DIR = "data" # Directory to save the model
MODEL_PATH = os.path.join(MODEL_DIR, "classification_pipeline.joblib")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_classifier():
    """
    Loads data, masks PII, trains a classifier, and saves the pipeline.
    """
    print(f"Loading data from {DATA_PATH}...")

    # --- Check if placeholder names were changed (They are changed now) ---
    # This check is technically redundant now but harmless
    if ACTUAL_EMAIL_COLUMN_NAME == "YOUR_EMAIL_COLUMN_HERE" or \
       ACTUAL_CATEGORY_COLUMN_NAME == "YOUR_CATEGORY_COLUMN_HERE":
        raise ValueError("Placeholder column names were not updated. This should not happen with the corrected code.")
    # --- End of check ---

    try:
        df = pd.read_csv(DATA_PATH)
        # Verify required columns exist using the names you provided
        if ACTUAL_EMAIL_COLUMN_NAME not in df.columns or ACTUAL_CATEGORY_COLUMN_NAME not in df.columns:
            raise ValueError(f"CSV must contain '{ACTUAL_EMAIL_COLUMN_NAME}' and '{ACTUAL_CATEGORY_COLUMN_NAME}' columns. Please check the column names in your CSV and update the script if needed.")
        print(f"Data loaded: {df.shape[0]} rows")

        # --- Handle potential NaN values in email or category columns ---
        # Check for NaNs before processing
        initial_rows = len(df)
        if df[ACTUAL_EMAIL_COLUMN_NAME].isnull().any():
            print(f"Warning: Found missing values in the '{ACTUAL_EMAIL_COLUMN_NAME}' column. Dropping rows with missing emails.")
            df.dropna(subset=[ACTUAL_EMAIL_COLUMN_NAME], inplace=True)
            print(f"Dropped {initial_rows - len(df)} rows due to missing emails.")
            initial_rows = len(df) # Update count

        if df[ACTUAL_CATEGORY_COLUMN_NAME].isnull().any():
            print(f"Warning: Found missing values in the '{ACTUAL_CATEGORY_COLUMN_NAME}' column. Dropping rows with missing categories.")
            df.dropna(subset=[ACTUAL_CATEGORY_COLUMN_NAME], inplace=True)
            print(f"Dropped {initial_rows - len(df)} rows due to missing categories.")


        # Convert category column to string just in case it's numerical
        df[ACTUAL_CATEGORY_COLUMN_NAME] = df[ACTUAL_CATEGORY_COLUMN_NAME].astype(str)
        # --- End of NaN handling ---

    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Please make sure 'combined_emails_with_natural_pii.csv' is in the project root directory.")
        return # Exit if file not found
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return # Exit on other read errors

    if len(df) == 0:
        print("Error: No data left after cleaning (NaN removal or empty file). Cannot train model.")
        return

    print("Applying PII masking to training data...")
    # Apply masking using the CORRECT email column name
    # Ensure the email column is string type before applying mask_pii
    try:
        df['masked_text'] = df[ACTUAL_EMAIL_COLUMN_NAME].astype(str).apply(lambda x: mask_pii(x)[0])
        print("PII masking complete.")
    except Exception as e:
        print(f"Error during PII masking: {e}")
        return # Exit if masking fails


    # Define features (masked text) and target (using the CORRECT category column name)
    X = df['masked_text']
    y = df[ACTUAL_CATEGORY_COLUMN_NAME]

    # Check if there are enough samples for stratification
    if len(df) == 0:
        print("Error: No data left after cleaning. Cannot train model.")
        return
    min_classes_for_stratify = 2 # Stratify usually needs at least 2 samples per class
    test_size = 0.2
    stratify_split = None # Default to None (no stratification)

    if len(y.unique()) < 2 :
         print(f"Warning: Only {len(y.unique())} unique classes found. Cannot stratify train/test split.")
         if len(df) < 5: # Need at least 5 samples to split reasonably without stratify
             test_size = 1/len(df) if len(df) > 1 else 0 # Avoid error if only 1 sample
             print(f"Very few samples ({len(df)}). Adjusting test size to {test_size}")
    elif any(y.value_counts() < min_classes_for_stratify):
        print("Warning: Some classes have fewer than 2 samples. Stratification is not possible. Proceeding without stratification.")
    else:
        # Only set stratify=y if conditions are met
        stratify_split = y
        print("Conditions met for stratified split.")


    # Split data
    try:
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_split)
    except ValueError as e:
         # This might happen if stratify=y was set but a class was still too small after potential splits
         print(f"Stratification failed during split: {e}. Proceeding without stratification.")
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    # Create a pipeline: TF-IDF Vectorizer -> Logistic Regression Classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)), # min_df=2 requires a word to appear in at least 2 docs
        ('clf', LogisticRegression(random_state=42, solver='liblinear', multi_class='auto', class_weight='balanced')) # Added class_weight='balanced'
    ])

    print("Training the classification model...")
    try:
        if len(X_train) == 0:
            print("Error: Training set is empty after split. Cannot train model.")
            return
        pipeline.fit(X_train, y_train)
        print("Training complete.")
    except ValueError as e:
        print(f"Error during model fitting: {e}")
        print("This might happen if the vocabulary is empty after applying TF-IDF settings (e.g., stop words, min_df). Check your data and TF-IDF parameters.")
        return


    # Evaluate the model only if test set is not empty
    if len(X_test) > 0 and len(y_test) > 0:
        print("\nEvaluating model on the test set:")
        try:
            y_pred = pipeline.predict(X_test)
            print(classification_report(y_test, y_pred, zero_division=0)) # Added zero_division=0
        except Exception as e:
            print(f"Could not generate classification report: {e}")
    else:
        print("\nTest set is empty. Skipping evaluation.")


    # Save the entire pipeline (vectorizer + classifier)
    print(f"Saving model pipeline to {MODEL_PATH}...")
    try:
        joblib.dump(pipeline, MODEL_PATH)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")


    return pipeline # Return trained pipeline

def load_classifier() -> Pipeline:
    """Loads the pre-trained classification pipeline."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Training a new one...")
        pipeline = train_classifier()
        if pipeline is None: # Handle case where training failed
             raise RuntimeError("Failed to train a new model.")
    else:
        print(f"Loading model from {MODEL_PATH}...")
        try:
            pipeline = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            print("Attempting to train a new model...")
            pipeline = train_classifier()
            if pipeline is None: # Handle case where training failed
                raise RuntimeError(f"Failed to load existing model and failed to train a new model.")
    return pipeline

def predict_category(masked_text: str, pipeline: Pipeline) -> str:
    """
    Predicts the category for a given masked email text using the loaded pipeline.

    Args:
        masked_text: The email text after PII masking.
        pipeline: The loaded scikit-learn pipeline.

    Returns:
        The predicted category label as a string.
    """
    try:
        # The pipeline expects an iterable (like a list), even for a single prediction
        prediction = pipeline.predict([masked_text])
        return prediction[0] # Return the first (and only) prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error predicting category" # Return a default error message


# --- Training Execution ---
# You can run this script directly to train the model: python models.py
if __name__ == "__main__":
    train_classifier()