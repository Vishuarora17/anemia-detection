import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.utils import resample

# --- IMPORTS for InceptionV3 ---
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
# ---

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
N_SPLITS = 5
EPOCHS = 30
FINE_TUNE_EPOCHS = 15

# --- 1. Load the dataset ---
df = pd.read_csv('anemia_dataset_cleaned.csv')
X = df['filepath']
y = df['label']

# --- K-Fold Cross-Validation Setup ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_no = 1
histories = []
all_scores = []
all_true_labels, all_pred_classes, all_pred_probs = [], [], []

# --- 2. Model Creation Function ---
def create_model():
    # --- Using InceptionV3 as the base model ---
    base_model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, base_model

# --- 3. K-Fold Training Loop ---
for train_index, test_index in skf.split(X, y):
    print(f"--- Starting InceptionV3 Fold {fold_no}/{N_SPLITS} ---")

    outer_train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    
    train_df, val_df = train_test_split(outer_train_df, test_size=0.1, random_state=42, stratify=outer_train_df['label'])

    df_majority = train_df[train_df.label == 'Non-Anemic']
    df_minority = train_df[train_df.label == 'Anemic']
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=30, width_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(dataframe=df_upsampled, x_col='filepath', y_col='label', target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=True)
    validation_generator = val_test_datagen.flow_from_dataframe(dataframe=val_df, x_col='filepath', y_col='label', target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)
    test_generator = val_test_datagen.flow_from_dataframe(dataframe=test_df, x_col='filepath', y_col='label', target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

    model, base_model = create_model()
    checkpoint = ModelCheckpoint(f'best_inception_fold_{fold_no}.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, callbacks=[checkpoint, early_stopping], verbose=0)
    
    # Fine-tuning stage for InceptionV3
    # We unfreeze the top two Inception blocks (from layer 249 onwards)
    base_model.trainable = True
    for layer in base_model.layers[:249]:
        layer.trainable = False
    for layer in base_model.layers[249:]:
        layer.trainable = True
        
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    history_fine_tune = model.fit(train_generator, epochs=FINE_TUNE_EPOCHS, validation_data=validation_generator, callbacks=[checkpoint, early_stopping], verbose=0)

    model.load_weights(f'best_inception_fold_{fold_no}.keras')
    
    full_history = {key: history.history[key] + history_fine_tune.history[key] for key in history.history}
    histories.append(full_history)

    print(f"--- Evaluating Fold {fold_no} ---")
    loss, accuracy = model.evaluate(test_generator)
    all_scores.append(accuracy)
    
    y_true_fold = test_generator.classes
    y_pred_probs_fold = model.predict(test_generator)
    y_pred_classes_fold = (y_pred_probs_fold > 0.5).astype("int32").flatten()
    
    all_true_labels.extend(y_true_fold)
    all_pred_probs.extend(y_pred_probs_fold)
    all_pred_classes.extend(y_pred_classes_fold)
    
    fold_no += 1

# --- Final Report for InceptionV3 ---
print("\n--- InceptionV3 Cross-Validation Complete ---")
print(f"Average Accuracy: {np.mean(all_scores)*100:.2f}% (+/- {np.std(all_scores)*100:.2f}%)")
mse = mean_squared_error(all_true_labels, all_pred_probs)
print(f"Aggregate Mean Squared Error (MSE): {mse:.4f}\n")

target_names = list(test_generator.class_indices.keys())
print("--- Aggregate Classification Report ---")
print(classification_report(all_true_labels, all_pred_classes, target_names=target_names))

print("--- Aggregate Confusion Matrix ---")
cm = confusion_matrix(all_true_labels, all_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Aggregate Confusion Matrix (InceptionV3 - All Folds)')
plt.savefig('confusion_matrix_inceptionv3.png')
print("\n Final InceptionV3 confusion matrix plot saved as 'confusion_matrix_inceptionv3.png'")
