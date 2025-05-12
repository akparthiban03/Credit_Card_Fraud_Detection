import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import os

def load_and_preprocess_data(file_path):
    # Load Dataset
    df = pd.read_csv(file_path)
    
    # Drop Unnecessary Columns
    columns_to_drop = [
        'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'trans_num',
        'street', 'city', 'state', 'zip', 'job', 'dob', 'merchant',
        'category', 'first', 'last', 'gender'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    print("Dataset shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    
    return df

def prepare_data(df):
    X = df.drop(columns=['is_fraud']).values
    y = df['is_fraud'].values.astype(np.int32)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Print class distribution
    print("\nOriginal class distribution:")
    print(np.bincount(y) / len(y))
    print("\nResampled class distribution:")
    print(np.bincount(y_resampled) / len(y_resampled))
    
    return X_resampled, y_resampled

def split_and_scale_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

def save_model_files(model, scaler, output_dir='model_files'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, 'fraud_detection_model.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Verify files exist
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Files saved and verified successfully")
    else:
        print("Error: Files not saved correctly")

def main():
    # Load and preprocess data
    df = load_and_preprocess_data(r"C:\Users\avine\Downloads\fraudTest.csv")
    
    # Prepare features and target
    X_resampled, y_resampled = prepare_data(df)
    
    # Split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X_resampled, y_resampled)
    
    # Build and train model
    model = build_model(X_train.shape[1])
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.3).astype(int)
    
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
    
    # Save model and scaler
    save_model_files(model, scaler)

if __name__ == "__main__":
    main()