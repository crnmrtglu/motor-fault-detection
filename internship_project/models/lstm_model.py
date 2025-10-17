
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import RobustScaler, LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras import
try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.optimizers import Adam
    from keras.utils import to_categorical
    print(" TensorFlow loaded!")
    print(f" GPU available: {tf.config.list_physical_devices('GPU')}")
except ImportError:
    print(" TensorFlow not installed. Please run 'pip install tensorflow'!")
    import sys
    sys.exit(1)

# Graphic settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("LSTM Motor Fault Detection")
print("="*50)

# Data Loading
train_data = pd.read_csv('internship_project/data/motor_training_data.csv')
test1_data = pd.read_csv('internship_project/data/motor_testing_data.csv')
test2_data = pd.read_csv('internship_project/data/motor_testing_data_6.csv')

df = pd.DataFrame(train_data)
print(f"Dataset: {df.shape}")

# Visualize class distribution
class_counts = df['state'].value_counts()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
class_counts.plot(kind='bar', color='skyblue')
plt.title('Motor Fault Class Distribution')
plt.xticks(rotation=45)
plt.ylabel('Sample Count')
plt.subplot(1, 2, 2)
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
plt.title('Class Proportions')
plt.tight_layout()
plt.savefig('lstm_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Label encoding
le = LabelEncoder()
df['state_encoded'] = le.fit_transform(df['state'])
print(f"Classes: {list(le.classes_)}")

# Feature Engineering - Time Series Features
def create_advanced_time_series_features(X):
    """Advanced time series features"""
    X_new = X.copy()
    
    # 1. Basic current features
    X_new['current_magnitude'] = np.sqrt(X['current_R']**2 + X['current_S']**2 + X['current_W']**2)
    X_new['current_mean'] = X[['current_R', 'current_S', 'current_W']].mean(axis=1)
    X_new['current_std'] = X[['current_R', 'current_S', 'current_W']].std(axis=1)
    X_new['current_imbalance'] = X_new['current_std'] / (X_new['current_mean'] + 1e-8)
    
    # 2. Advanced current features
    X_new['current_max'] = X[['current_R', 'current_S', 'current_W']].max(axis=1)
    X_new['current_min'] = X[['current_R', 'current_S', 'current_W']].min(axis=1)
    X_new['current_range'] = X_new['current_max'] - X_new['current_min']
    X_new['current_rms'] = np.sqrt((X['current_R']**2 + X['current_S']**2 + X['current_W']**2) / 3)

    # 3. Vibration features
    X_new['vibration_magnitude'] = np.sqrt(X['acceleration_X']**2 + X['acceleration_Y']**2 + X['acceleration_Z']**2)
    X_new['vibration_mean'] = X[['acceleration_X', 'acceleration_Y', 'acceleration_Z']].mean(axis=1)
    X_new['vibration_std'] = X[['acceleration_X', 'acceleration_Y', 'acceleration_Z']].std(axis=1)
    X_new['vibration_rms'] = np.sqrt((X['acceleration_X']**2 + X['acceleration_Y']**2 + X['acceleration_Z']**2) / 3)

    # 4. Current-vibration interactions
    X_new['current_vibration_ratio'] = X_new['current_magnitude'] / (X_new['vibration_magnitude'] + 1e-8)
    X_new['power_factor'] = X_new['current_rms'] * X_new['vibration_rms']
    X_new['efficiency_indicator'] = X_new['current_mean'] / (X_new['vibration_mean'] + 1e-8)

    # 5. Phase features (critical for 3-phase motors)
    X_new['phase_R_dominance'] = X['current_R'] / (X_new['current_mean'] + 1e-8)
    X_new['phase_S_dominance'] = X['current_S'] / (X_new['current_mean'] + 1e-8)
    X_new['phase_W_dominance'] = X['current_W'] / (X_new['current_mean'] + 1e-8)
    
    return X_new

print("🔧 Advanced Time Series Feature Engineering...")
X = df.drop(columns=['state', 'state_encoded'], axis=1)
y = df['state_encoded'].values

X_engineered = create_advanced_time_series_features(X)
print(f" Features: {X.shape[1]} → {X_engineered.shape[1]}")

# Feature correlation
plt.figure(figsize=(10, 8))
correlation_matrix = X_engineered.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('LSTM Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('lstm_feature_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# Sequence creation
def create_sequences(X, y, sequence_length=20):
    """Create time series sequences"""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])  # Last timestep's label
    
    return np.array(X_seq), np.array(y_seq)

# Normalize the data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_engineered)

# Sequence length 
SEQUENCE_LENGTH = 25  
print(f" Sequence Length: {SEQUENCE_LENGTH}")

# Create sequences
print(" Creating LSTM sequences...")
X_sequences, y_sequences = create_sequences(X_scaled, y, SEQUENCE_LENGTH)
print(f" Sequences: {X_sequences.shape}")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
)

print(f" Train: {X_train.shape}, Validation: {X_val.shape}")

# Class distribution balancing
print(" Class balancing...")
unique, counts = np.unique(y_train, return_counts=True)
class_weights = {}
max_count = max(counts)
for i, count in enumerate(counts):
    class_weights[i] = max_count / count

print("Class weights:", {le.inverse_transform([k])[0]: f"{v:.2f}" for k, v in class_weights.items()})

# Balancing plot
original_dist = Counter(y_train)
plt.figure(figsize=(10, 5))
labels = [le.inverse_transform([x])[0] for x in original_dist.keys()]
plt.bar(labels, original_dist.values(), color='lightcoral', alpha=0.7)
plt.title('LSTM Training Set Distribution')
plt.xticks(rotation=45)
plt.ylabel('Sample Count')
plt.tight_layout()
plt.savefig('lstm_class_distribution_train.png', dpi=300, bbox_inches='tight')
plt.show()

# One-hot encoding for LSTM
n_classes = len(le.classes_)
y_train_onehot = to_categorical(y_train, num_classes=n_classes)
y_val_onehot = to_categorical(y_val, num_classes=n_classes)

# LSTM Model
print(" Creating LSTM model...")
model = Sequential([
    #
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
    Dropout(0.3),

    # Second LSTM layer
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    
    # Dense layers 
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])


optimizer = Adam(learning_rate=0.002) 

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Architecture:")
model.summary()

# Model training
print("LSTM training is starting...")

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Learning rate 
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=15,  
    batch_size=64,  # Smaller batch‚
    class_weight=class_weights,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Training history plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Validation predictions
print(" LSTM Validation results:")
y_pred_proba = model.predict(X_val)
y_pred = np.argmax(y_pred_proba, axis=1)

# make labels inverse
y_val_labels = le.inverse_transform(y_val)
y_pred_labels = le.inverse_transform(y_pred)

val_acc = accuracy_score(y_val_labels, y_pred_labels)
val_f1 = f1_score(y_val_labels, y_pred_labels, average='weighted')

print(f" LSTM accuracy: {val_acc:.4f}")
print(f" LSTM Weighted F1: {val_f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_val_labels, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('LSTM Confusion Matrix - Validation')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('lstm_confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed class performance
print(f"\n LSTM Class-Based Performance:")
print(classification_report(y_val_labels, y_pred_labels, digits=4))

# create sequences for test sets
def evaluate_test_set(test_data, set_name):
    test_df = pd.DataFrame(test_data)
    X_test = test_df.drop(columns=['state'], axis=1)
    y_test = test_df['state']
    
    # Feature engineering
    X_test_eng = create_advanced_time_series_features(X_test)
    X_test_scaled = scaler.transform(X_test_eng)

    # create sequences
    y_test_encoded = le.transform(y_test)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_encoded, SEQUENCE_LENGTH)
    
    if len(X_test_seq) == 0:
        print(f" {set_name}: Sequence could not be created!")
        return 0, 0
    
    # Predict
    y_test_pred_proba = model.predict(X_test_seq)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    
    # Labels
    y_test_labels = le.inverse_transform(y_test_seq)
    y_test_pred_labels = le.inverse_transform(y_test_pred)
    
    acc = accuracy_score(y_test_labels, y_test_pred_labels)
    f1 = f1_score(y_test_labels, y_test_pred_labels, average='weighted')
    
    print(f"\n LSTM {set_name}:")
    print(f" Accuracy: {acc:.4f}")
    print(f" F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_labels, y_test_pred_labels, labels=le.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'LSTM Confusion Matrix - {set_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'lstm_confusion_matrix_{set_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return acc, f1

# evaluate test sets
test1_acc, test1_f1 = evaluate_test_set(test1_data, "Test Set 1")
test2_acc, test2_f1 = evaluate_test_set(test2_data, "Test Set 2")

# Performance comparison
scores_comparison = {
    'Dataset': ['Validation', 'Test Set 1', 'Test Set 2'],
    'Accuracy': [val_acc, test1_acc, test2_acc],
    'F1-Score': [val_f1, test1_f1, test2_f1]
}
comparison_df = pd.DataFrame(scores_comparison)

plt.figure(figsize=(10, 6))
x = np.arange(len(comparison_df['Dataset']))
width = 0.35
plt.bar(x - width/2, comparison_df['Accuracy'], width, label='Accuracy', alpha=0.8)
plt.bar(x + width/2, comparison_df['F1-Score'], width, label='F1-Score', alpha=0.8)
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.title('LSTM Model Performance Comparison')
plt.xticks(x, comparison_df['Dataset'])
plt.legend()
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Final summary
print(f"\n LSTM FINAL SUMMARY")
print("="*40)
print(f"📊 Validation F1: {val_f1:.4f}")
print(f"📊 Test1 F1: {test1_f1:.4f}")
print(f"📊 Test2 F1: {test2_f1:.4f}")

all_scores = [val_f1, test1_f1, test2_f1]
mean_score = np.mean([test1_f1,test2_f1])
std_score = np.std(all_scores)

print(f"📊 Average F1: {mean_score:.4f}")
print(f"📊 Std Dev: {std_score:.4f}")

if mean_score >= 0.8:
    print(" %80+ F1-Score!")
elif mean_score >= 0.7:
    print("Good performance! %70+ F1-Score")
elif mean_score >= 0.6:
    print("Average performance! %60+ F1-Score")
else:
    print("Low performance - improvement needed")

print(f"\n💾 LSTM Saved Graphs:")
print("  📊 lstm_class_distribution.png")
print("  🔗 lstm_feature_correlation.png")
print("  📈 lstm_class_distribution_train.png")
print("  📉 lstm_training_history.png")
print("  📋 lstm_confusion_matrix_validation.png")
print("  📋 lstm_confusion_matrix_test_set_1.png")
print("  📋 lstm_confusion_matrix_test_set_2.png")
print("  📊 lstm_performance_comparison.png")

print("\n LSTM motor fault detection completed!")