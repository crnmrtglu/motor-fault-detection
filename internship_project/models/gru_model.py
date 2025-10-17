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
    from keras.layers import GRU, Dense, Dropout
    from keras.optimizers import Adam
    from keras.utils import to_categorical
    print(" TensorFlow loaded!")
    # for gpu acceleration
    tf.config.experimental.set_memory_growth = True
except ImportError:
    print(" TensorFlow not installed!")
    import sys
    sys.exit(1)

# graphic settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print(" GRU Motor Fault Detection")
print("="*45)

# Data Loading
train_data = pd.read_csv('internship_project/data/motor_training_data.csv')
test1_data = pd.read_csv('internship_project/data/motor_testing_data.csv')
test2_data = pd.read_csv('internship_project/data/motor_testing_data_6.csv')

df = pd.DataFrame(train_data)
print(f" Dataset: {df.shape}")

#  graphic: Class distribution
class_counts = df['state'].value_counts()
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
class_counts.plot(kind='bar', color='lightblue')
plt.title('Motor Fault Distribution')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
plt.title('Class Ratios')
plt.tight_layout()
plt.savefig('gru_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


le = LabelEncoder()
df['state_encoded'] = le.fit_transform(df['state'])
print(f" Classes: {list(le.classes_)}")

#  Feature Engineering 
def create_fast_features(X):
    """feature engineering"""
    X_new = X.copy()
    
    
    X_new['current_magnitude'] = np.sqrt(X['current_R']**2 + X['current_S']**2 + X['current_W']**2)
    X_new['current_imbalance'] = X[['current_R', 'current_S', 'current_W']].std(axis=1)
    X_new['current_mean'] = X[['current_R', 'current_S', 'current_W']].mean(axis=1)


    X_new['vibration_magnitude'] = np.sqrt(X['acceleration_X']**2 + X['acceleration_Y']**2 + X['acceleration_Z']**2)
    X_new['vibration_std'] = X[['acceleration_X', 'acceleration_Y', 'acceleration_Z']].std(axis=1)

   
    X_new['current_vibration_ratio'] = X_new['current_magnitude'] / (X_new['vibration_magnitude'] + 1e-8)

    
    X_new['efficiency_proxy'] = X_new['current_mean'] / (X_new['vibration_magnitude'] + 1e-8)
    
    return X_new

print(" Feature Engineering...")
X = df.drop(columns=['state', 'state_encoded'], axis=1)
y = df['state_encoded'].values

X_engineered = create_fast_features(X)
print(f" Features: {X.shape[1]} → {X_engineered.shape[1]} ")

#  correlation plot
plt.figure(figsize=(8, 6))
corr_matrix = X_engineered.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('GRU Feature Correlations')
plt.tight_layout()
plt.savefig('gru_feature_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# Better preprocessing
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_engineered)

# Longer sequence - better pattern capture
SEQUENCE_LENGTH = 20  
print(f" Learning Sequence Length: {SEQUENCE_LENGTH}")

# better sequence creation  - more data
def create_better_sequences(X, y, seq_len):
    """Better sequence creation for more data"""
    # Take every 2nd sample 
    step = 2
    X_seq, y_seq = [], []
    
    for i in range(0, len(X) - seq_len + 1, step):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    
    return np.array(X_seq), np.array(y_seq)

print(" Better sequence creation for learning...")
X_sequences, y_sequences = create_better_sequences(X_scaled, y, SEQUENCE_LENGTH)
print(f" Better Learning Sequences: {X_sequences.shape}")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
)

print(f" Train: {X_train.shape}, Val: {X_val.shape}")

# Class weights 
unique, counts = np.unique(y_train, return_counts=True)
class_weights = {i: max(counts)/count for i, count in enumerate(counts)}

# Sampling distribution plot
plt.figure(figsize=(8, 4))
train_dist = Counter(y_train)
labels = [le.inverse_transform([x])[0] for x in train_dist.keys()]
plt.bar(labels, train_dist.values(), color='lightgreen', alpha=0.7)
plt.title('GRU Training Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('gru_train_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# One-hot encoding
n_classes = len(le.classes_)
y_train_onehot = to_categorical(y_train, num_classes=n_classes)
y_val_onehot = to_categorical(y_val, num_classes=n_classes)

#  GRU Model 
model = Sequential([
    # double GRU layer 
    GRU(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
    Dropout(0.3),
    GRU(64, return_sequences=False),
    Dropout(0.3),

    # powerful dense layers
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])

# Optimize learning rate 
optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("📋 Fast GRU Architecture:")
model.summary()

# Better training for higher performance
print(" Better training for higher performance...")

# advanced callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_f1_score',  # F1 score monitor
    patience=5,  # More patience
    restore_best_weights=True,
    mode='max'  # Maximize F1
)

# Learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_f1_score',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    mode='max'
)

# F1 metric addition
from tensorflow.keras import backend as K

def f1_score_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2*((precision_val*recall_val)/(precision_val+recall_val+K.epsilon()))

# Model compile with F1 metric
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', f1_score_metric]
)

history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=25,  
    batch_size=64,  # Smaller batch size - more sensitivity
    class_weight=class_weights,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# Training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='Train Loss')
plt.plot(history.history['val_loss'], 'r-', label='Val Loss')
plt.title('GRU Fast Training Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='Train Acc')
plt.plot(history.history['val_accuracy'], 'r-', label='Val Acc')
plt.title('GRU Fast Training Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('gru_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Validation results
print("GRU Results:")

y_pred_proba = model.predict(X_val)
y_pred = np.argmax(y_pred_proba, axis=1)

# Convert back to labels
y_val_labels = le.inverse_transform(y_val)
y_pred_labels = le.inverse_transform(y_pred)

val_acc = accuracy_score(y_val_labels, y_pred_labels)
val_f1 = f1_score(y_val_labels, y_pred_labels, average='weighted')

print(f" GRU Accuracy: {val_acc:.4f}")
print(f" GRU Weighted F1: {val_f1:.4f}")
model_report = classification_report(y_val_labels, y_pred_labels, target_names=le.classes_)
print(model_report)

# Fast confusion matrix
cm = confusion_matrix(y_val_labels, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('GRU Confusion Matrix - Validation')
plt.tight_layout()
plt.savefig('gru_confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
plt.show()

# Test sets - quick evaluation
def fast_test_evaluation(test_data, set_name):
    test_df = pd.DataFrame(test_data)
    X_test = test_df.drop(columns=['state'], axis=1)
    y_test = test_df['state']
    
    # Fast preprocessing
    X_test_eng = create_fast_features(X_test)
    X_test_scaled = scaler.transform(X_test_eng)
    
    # Better sequences for testing
    y_test_encoded = le.transform(y_test)
    X_test_seq, y_test_seq = create_better_sequences(X_test_scaled, y_test_encoded, SEQUENCE_LENGTH)
    
    if len(X_test_seq) == 0:
        return 0, 0
    
    # Predict
    y_test_pred_proba = model.predict(X_test_seq)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    
    # Labels
    y_test_labels = le.inverse_transform(y_test_seq)
    y_test_pred_labels = le.inverse_transform(y_test_pred)
    
    acc = accuracy_score(y_test_labels, y_test_pred_labels)
    f1 = f1_score(y_test_labels, y_test_pred_labels, average='weighted')
    
    print(f"\n GRU {set_name}:")
    print(f" Accuracy: {acc:.4f}")
    print(f" F1-Score: {f1:.4f}")
    
    # Fast confusion matrix
    cm = confusion_matrix(y_test_labels, y_test_pred_labels, labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'GRU Fast Confusion Matrix - {set_name}')
    plt.tight_layout()
    plt.savefig(f'gru_confusion_matrix_{set_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return acc, f1

# Test evaluations
test1_acc, test1_f1 = fast_test_evaluation(test1_data, "Test Set 1")
test2_acc, test2_f1 = fast_test_evaluation(test2_data, "Test Set 2")

# Fast performance comparison
comparison_data = {
    'Dataset': ['Validation', 'Test Set 1', 'Test Set 2'],
    'Accuracy': [val_acc, test1_acc, test2_acc],
    'F1-Score': [val_f1, test1_f1, test2_f1]
}

plt.figure(figsize=(8, 5))
x = np.arange(len(comparison_data['Dataset']))
width = 0.35
plt.bar(x - width/2, comparison_data['Accuracy'], width, label='Accuracy', alpha=0.8)
plt.bar(x + width/2, comparison_data['F1-Score'], width, label='F1-Score', alpha=0.8)
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.title('GRU Fast Performance Comparison')
plt.xticks(x, comparison_data['Dataset'])
plt.legend()
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('gru_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Final summary
print(f"\n GRU SUMMARY")
print("="*35)
print(f" Validation F1: {val_f1:.4f}")
print(f" Test1 F1: {test1_f1:.4f}")
print(f" Test2 F1: {test2_f1:.4f}")

all_scores = [val_f1, test1_f1, test2_f1]
mean_score = np.mean([test1_f1, test2_f1])
std_score = np.std(all_scores)

print(f" Mean F1: {mean_score:.4f}")
print(f" Std Dev: {std_score:.4f}")

if mean_score >= 0.75:
    print(" Excellent GRU Performance!")
elif mean_score >= 0.65:
    print(" Good GRU Performance!")
elif mean_score >= 0.55:
    print(" Decent GRU Performance")
else:
    print(" GRU needs improvement")

print(f"\n💾 GRU Graphics:")
print("  📊 gru_class_distribution.png")
print("  🔗 gru_feature_correlation.png")
print("  📈 gru_train_distribution.png")
print("  📉 gru_training_history.png")
print("  📋 gru_confusion_matrix_validation.png")
print("  📋 gru_confusion_matrix_test_set_1.png")
print("  📋 gru_confusion_matrix_test_set_2.png")
print("  📊 gru_performance_comparison.png")

print(f"\n GRU training completed!")