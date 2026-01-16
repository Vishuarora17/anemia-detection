import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

# --- 1. Load Data and Create a Final, Held-out Test Set ---
df = pd.read_csv('anemia_dataset_cleaned.csv')
# This test set was NOT used during the K-Fold training of any model
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"Using {len(test_df)} images for final ensemble evaluation.")

# --- 2. Load Your Three Trained Models ---
print("Loading all three trained models...")
try:
    # Assumes the best model from the first fold of each K-Fold run is used
    model_densenet = tf.keras.models.load_model('best_densenet_fold_1.keras')
    model_vgg16 = tf.keras.models.load_model('best_vgg16_fold_1.keras')
    model_inception = tf.keras.models.load_model('best_inception_fold_1.keras')
    print(" All three models loaded successfully!")
except Exception as e:
    print(f" Error loading models: {e}")
    print("Please make sure you have run all three K-Fold scripts to generate the saved '.keras' files first.")
    exit()
    
# --- 3. Create Separate Data Generators for Each Model's Preprocessing ---
# Each model requires its own specific preprocessing function
test_datagen_densenet = ImageDataGenerator(preprocessing_function=densenet_preprocess)
test_datagen_vgg16 = ImageDataGenerator(preprocessing_function=vgg16_preprocess)
test_datagen_inception = ImageDataGenerator(preprocessing_function=inception_preprocess)

test_generator_densenet = test_datagen_densenet.flow_from_dataframe(
    dataframe=test_df, x_col='filepath', y_col='label',
    target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)
test_generator_vgg16 = test_datagen_vgg16.flow_from_dataframe(
    dataframe=test_df, x_col='filepath', y_col='label',
    target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)
test_generator_inception = test_datagen_inception.flow_from_dataframe(
    dataframe=test_df, x_col='filepath', y_col='label',
    target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)

# --- 4. Get Predictions from Each Model ---
print("\nGetting predictions from each model...")
preds_densenet = model_densenet.predict(test_generator_densenet)
preds_vgg16 = model_vgg16.predict(test_generator_vgg16)
preds_inception = model_inception.predict(test_generator_inception)

# --- 5. Ensemble the Predictions by Averaging ---
print("Averaging predictions to create ensemble result...")
ensemble_probs = (preds_densenet + preds_vgg16 + preds_inception) / 3.0

# Get final class predictions and true labels
y_true = test_generator_densenet.classes
ensemble_classes = (ensemble_probs > 0.5).astype("int32").flatten()

# --- 6. Evaluate the Final Ensemble Performance ---
print("\n--- Final Ensemble Performance Report ---")

# MSE
mse = mean_squared_error(y_true, ensemble_probs)
print(f"Aggregate Mean Squared Error (MSE): {mse:.4f}\n")

# Classification Report
target_names = list(test_generator_densenet.class_indices.keys())
report = classification_report(y_true, ensemble_classes, target_names=target_names)
print("--- Aggregate Classification Report ---")
print(report)

# Confusion Matrix
print("--- Aggregate Confusion Matrix ---")
cm = confusion_matrix(y_true, ensemble_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Final Ensemble Confusion Matrix (DenseNet121 + VGG16 + InceptionV3)')
plt.savefig('confusion_matrix_final_ensemble.png')
print("\n Final ensemble confusion matrix saved as 'confusion_matrix_final_ensemble.png'")
