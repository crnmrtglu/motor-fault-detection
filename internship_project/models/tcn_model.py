
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Input, Concatenate, LayerNormalization, SpatialDropout1D, Add, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.regularizers import l2

print(" TCN Motor Fault Detection")
print("="*50)

# data loading
train_data = pd.read_csv('internship_project/data/motor_training_data.csv')
test1_data = pd.read_csv('internship_project/data/motor_testing_data.csv')
test2_data = pd.read_csv('internship_project/data/motor_testing_data_6.csv')

df = pd.DataFrame(train_data)
print(f"Dataset: {df.shape}")

# Label encoding
le = LabelEncoder()
df['state_encoded'] = le.fit_transform(df['state'])
print(f" Classes: {list(le.classes_)}")

# Feature Engineering
def create_motor_features(X):
    """Motor-specific feature engineering"""
    X_new = X.copy()
    
    # Most critical current features
    X_new['current_magnitude'] = np.sqrt(X['current_R']**2 + X['current_S']**2 + X['current_W']**2)
    X_new['current_imbalance'] = X[['current_R', 'current_S', 'current_W']].std(axis=1)
    X_new['current_mean'] = X[['current_R', 'current_S', 'current_W']].mean(axis=1)

    # Most critical vibration features
    X_new['vibration_magnitude'] = np.sqrt(X['acceleration_X']**2 + X['acceleration_Y']**2 + X['acceleration_Z']**2)
    X_new['vibration_std'] = X[['acceleration_X', 'acceleration_Y', 'acceleration_Z']].std(axis=1)

    # Most important interaction
    X_new['current_vibration_ratio'] = X_new['current_magnitude'] / (X_new['vibration_magnitude'] + 1e-8)

    # Motor efficiency proxy
    X_new['efficiency_proxy'] = X_new['current_mean'] / (X_new['vibration_magnitude'] + 1e-8)
    
    return X_new

# Feature engineering
X = df.drop(columns=['state', 'state_encoded'])
y = df['state_encoded'].values
X_engineered = create_motor_features(X)
print(f"Features: {X.shape[1]} → {X_engineered.shape[1]}")

# Feature selection
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_engineered, y)
feature_importance = pd.DataFrame({
    'feature': X_engineered.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

top_features = feature_importance.head(15)['feature'].tolist()
X_selected = X_engineered[top_features]

# Sequence creation
def create_sequences(X, y, seq_len=20):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

# Data preparation
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)
X_sequences, y_sequences = create_sequences(X_scaled, y, 20)

X_train, X_val, y_train, y_val = train_test_split(
    X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
)

# Class weights
unique, counts = np.unique(y_train, return_counts=True)
class_weights = {i: np.sqrt(len(y_train) / (len(unique) * count)) 
                for i, count in enumerate(counts)}

# One-hot encoding
n_classes = len(le.classes_)
y_train_onehot = to_categorical(y_train, n_classes)
y_val_onehot = to_categorical(y_val, n_classes)

# TCN Block
def tcn_block(x, filters, kernel_size, dilation_rate, dropout=0.2):
    conv1 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
                   padding='causal', kernel_regularizer=l2(0.001))(x)
    conv1 = LayerNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = SpatialDropout1D(dropout)(conv1)
    
    conv2 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
                   padding='causal', kernel_regularizer=l2(0.001))(conv1)
    conv2 = LayerNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = SpatialDropout1D(dropout)(conv2)
    
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding='same')(x)
    
    return Add()([x, conv2])

# Model
def create_tcn_model(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    
    # TCN layers
    x = tcn_block(inputs, 64, 3, 1, 0.2)
    x = tcn_block(x, 64, 3, 2, 0.2)
    x = tcn_block(x, 96, 3, 4, 0.25)
    x = tcn_block(x, 128, 3, 8, 0.3)
    
    # Pooling
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    pooled = Concatenate()([max_pool, avg_pool])
    
    # Classification
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(pooled)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# Model compilation
model = create_tcn_model((20, len(top_features)), n_classes)
model.compile(
    optimizer=Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Model parameters: {model.count_params():,}")

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-6)
]

# Training
print(" Training begins...")
history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=20,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Validation results
y_pred = np.argmax(model.predict(X_val), axis=1)
y_val_labels = le.inverse_transform(y_val)
y_pred_labels = le.inverse_transform(y_pred)

val_f1 = f1_score(y_val_labels, y_pred_labels, average='weighted')
print(f" Validation F1: {val_f1:.4f}")

# Test evaluation
def evaluate_test(test_data, name):
    test_df = pd.DataFrame(test_data)
    X_test = test_df.drop(columns=['state'])
    y_test = test_df['state']
    
    X_test_eng = create_motor_features(X_test)[top_features]
    X_test_scaled = scaler.transform(X_test_eng)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, le.transform(y_test), 20)
    
    if len(X_test_seq) == 0:
        return 0
    
    y_test_pred = np.argmax(model.predict(X_test_seq), axis=1)
    test_f1 = f1_score(le.inverse_transform(y_test_seq), 
                      le.inverse_transform(y_test_pred), average='weighted')
    print(f" {name} F1: {test_f1:.4f}")
    return test_f1

test1_f1 = evaluate_test(test1_data, "Test Set 1")
test2_f1 = evaluate_test(test2_data, "Test Set 2")

# Final results
mean_f1 = np.mean([test1_f1, test2_f1])
print(f"\n mean F1: {mean_f1:.4f}")

if mean_f1 >= 0.8:
    print("🎉 %80+ F1-Score!")
elif mean_f1 >= 0.75:
    print("✅  %75+ F1-Score")
else:
    print("needs improvement")
    # Classification reports
    print("\nValidation Classification Report:")
    print(classification_report(y_val_labels, y_pred_labels))

    # Test set classification reports
    def print_test_classification_report(test_data, name):
        test_df = pd.DataFrame(test_data)
        X_test = test_df.drop(columns=['state'])
        y_test = test_df['state']
        
        X_test_eng = create_motor_features(X_test)[top_features]
        X_test_scaled = scaler.transform(X_test_eng)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, le.transform(y_test), 20)
        
        if len(X_test_seq) == 0:
            return
        
        y_test_pred = np.argmax(model.predict(X_test_seq), axis=1)
        y_test_labels = le.inverse_transform(y_test_seq)
        y_test_pred_labels = le.inverse_transform(y_test_pred)
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test_labels, y_test_pred_labels))

    print_test_classification_report(test1_data, "Test Set 1")
    print_test_classification_report(test2_data, "Test Set 2")

# Class distribution
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
plt.savefig('tcn_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature correlation (top features)
plt.figure(figsize=(10, 8))
correlation_matrix = X_selected.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('TCN Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('tcn_feature_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# Training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('TCN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('TCN Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tcn_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Validation confusion matrix
cm_val = confusion_matrix(y_val_labels, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('TCN Confusion Matrix - Validation')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('tcn_confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
plt.show()

# Test evaluation with confusion matrices
def evaluate_test_detailed(test_data, name):
    test_df = pd.DataFrame(test_data)
    X_test = test_df.drop(columns=['state'])
    y_test = test_df['state']
    
    X_test_eng = create_motor_features(X_test)[top_features]
    X_test_scaled = scaler.transform(X_test_eng)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, le.transform(y_test), 20)
    
    if len(X_test_seq) == 0:
        return 0
    
    y_test_pred = np.argmax(model.predict(X_test_seq), axis=1)
    y_test_labels = le.inverse_transform(y_test_seq)
    y_test_pred_labels = le.inverse_transform(y_test_pred)
    
    test_f1 = f1_score(y_test_labels, y_test_pred_labels, average='weighted')
    print(f" {name} F1: {test_f1:.4f}")
    
    # Confusion matrix
    cm_test = confusion_matrix(y_test_labels, y_test_pred_labels, labels=le.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'TCN Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'tcn_confusion_matrix_{name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_f1

test1_f1 = evaluate_test_detailed(test1_data, "Test Set 1")
test2_f1 = evaluate_test_detailed(test2_data, "Test Set 2")

# Performance comparison
scores_comparison = {
    'Dataset': ['Validation', 'Test Set 1', 'Test Set 2'],
    'F1-Score': [val_f1, test1_f1, test2_f1]
}
comparison_df = pd.DataFrame(scores_comparison)

plt.figure(figsize=(10, 6))
plt.bar(comparison_df['Dataset'], comparison_df['F1-Score'], 
        color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
plt.xlabel('Dataset')
plt.ylabel('F1-Score')
plt.title('TCN Model Performance Comparison')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(comparison_df['F1-Score']):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('tcn_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(comparison_df['Dataset'], comparison_df['F1-Score'], 
        color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
plt.xlabel('Dataset')
plt.ylabel('F1-Score')
plt.title('TCN Model Performance Comparison')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)


print(f"\n💾 TCN Saved plots:")
print("  📊 tcn_class_distribution.png")
print("  🔗 tcn_feature_correlation.png") 
print("  📈 tcn_training_history.png")
print("  📋 tcn_confusion_matrix_validation.png")
print("  📋 tcn_confusion_matrix_test_set_1.png")
print("  📋 tcn_confusion_matrix_test_set_2.png")
print("  📊 tcn_performance_comparison.png")

