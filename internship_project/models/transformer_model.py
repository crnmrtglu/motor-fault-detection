import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import RobustScaler, LabelEncoder
from collections import Counter
import warnings
from imblearn.combine import SMOTEENN
warnings.filterwarnings('ignore')

# TensorFlow/Keras import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    print(" TensorFlow loaded!")
    print(f" GPU available: {len(tf.config.list_physical_devices('GPU'))} GPU")
except ImportError:
    print(" TensorFlow not installed!")
    import sys
    sys.exit(1)

# Plot settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print(" TRANSFORMER Motor Fault Detection ")
print("="*55)

# Data loading
train_data = pd.read_csv('internship_project/data/motor_training_data.csv')
test1_data = pd.read_csv('internship_project/data/motor_testing_data.csv')
test2_data = pd.read_csv('internship_project/data/motor_testing_data_6.csv')

df = pd.DataFrame(train_data)
print(f" Dataset: {df.shape}")

# Class distribution visualization  
class_counts = df['state'].value_counts()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
class_counts.plot(kind='bar', color='gold')
plt.title('Transformer - Motor Fault Distribution')
plt.xticks(rotation=45)
plt.ylabel('Sample Count')
plt.subplot(1, 2, 2)
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
plt.title('Class Proportions')
plt.tight_layout()
plt.savefig('transformer_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Label encoding
le = LabelEncoder()
df['state_encoded'] = le.fit_transform(df['state'])
print(f"🏷️ Classes: {list(le.classes_)}")

# Advanced Feature Engineering for Transformer

#  Feature Engineering 
def create_transformer_features(X):
    """feature engineering"""
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

print(" Advanced Feature Engineering for Transformer...")
X = df.drop(columns=['state', 'state_encoded'], axis=1)
y = df['state_encoded'].values

X_engineered = create_transformer_features(X)
print(f" Features: {X.shape[1]} → {X_engineered.shape[1]} (rich feature set)")

# Feature correlation
plt.figure(figsize=(15, 12))
correlation_matrix = X_engineered.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Transformer Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('transformer_feature_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# Preprocessing
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_engineered)


# optimal sequence length for transformer
SEQUENCE_LENGTH = 15  
print(f" Optimized Transformer Sequence Length: {SEQUENCE_LENGTH}")

# create transformer_sequences function
def create_transformer_sequences(X, y, seq_len):
    step = 3 
    X_seq, y_seq = [], []
    
    for i in range(0, len(X) - seq_len + 1, step):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    
    return np.array(X_seq), np.array(y_seq)

print(" Creating Transformer sequences...")
X_sequences, y_sequences = create_transformer_sequences(X_scaled, y, SEQUENCE_LENGTH)
print(f" Transformer Sequences: {X_sequences.shape}")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
)

print(f" Train: {X_train.shape}, Val: {X_val.shape}")

# Class weights
unique, counts = np.unique(y_train, return_counts=True)
class_weights = {i: max(counts)/count for i, count in enumerate(counts)}

# Training distribution
plt.figure(figsize=(10, 5))
train_dist = Counter(y_train)
labels = [le.inverse_transform([x])[0] for x in train_dist.keys()]
plt.bar(labels, train_dist.values(), color='gold', alpha=0.7)
plt.title('Transformer Training Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('transformer_train_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# One-hot encoding
n_classes = len(le.classes_)
y_train_onehot = to_categorical(y_train, num_classes=n_classes)
y_val_onehot = to_categorical(y_val, num_classes=n_classes)


def create_transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Simplified Transformer block"""
    # Multi-head self-attention 
    attention_layer = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=head_size,
        dropout=dropout
    )
    attention_output = attention_layer(inputs, inputs)
    
    # Add & Norm
    attention_output = Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Basit Feed forward network
    ffn = tf.keras.Sequential([
        Dense(ff_dim, activation="relu"),
        Dropout(dropout),
        Dense(inputs.shape[-1]),
    ])
    ffn_output = ffn(attention_output)
    
    # Add & Norm
    ffn_output = Dropout(dropout)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)

def build_simple_transformer_model(input_shape, num_classes):
    """simplified but powerful Transformer model"""
    inputs = Input(shape=input_shape)
    
    
    x = Dense(64)(inputs)  

    #first block
    x = create_transformer_block(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.2)

    # second block - smaller
    x = create_transformer_block(x, head_size=16, num_heads=2, ff_dim=64, dropout=0.3)

    # Global pooling and classification
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    
    # more simple classification head
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs, outputs)

print(" Transformer Model...")
model = build_simple_transformer_model((SEQUENCE_LENGTH, X_train.shape[2]), n_classes)


optimizer = Adam(learning_rate=0.002)  

# Compile with F1 tracking
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("📋 Transformer Architecture:")
model.summary()

# training
print("Transformer training...")

# aggressive early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Accuracy focus
    patience=4,  # Less patience
    restore_best_weights=True,
    mode='max'
)

# aggressive learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5, 
    patience=2,  
    min_lr=1e-6,
    mode='max'
)

history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=20,  
    batch_size=64,  
    class_weight=class_weights,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# Training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='Train Loss')
plt.plot(history.history['val_loss'], 'r-', label='Val Loss')
plt.title('Transformer Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='Train Acc')
plt.plot(history.history['val_accuracy'], 'r-', label='Val Acc')
plt.title('Transformer Training Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('transformer_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Validation results
print(" Transformer Results:")
y_pred_proba = model.predict(X_val)
y_pred = np.argmax(y_pred_proba, axis=1)

# Convert back to labels
y_val_labels = le.inverse_transform(y_val)
y_pred_labels = le.inverse_transform(y_pred)

val_acc = accuracy_score(y_val_labels, y_pred_labels)
val_f1 = f1_score(y_val_labels, y_pred_labels, average='weighted')

print(f"🎯 Transformer Accuracy: {val_acc:.4f}")
print(f"📊 Transformer Weighted F1: {val_f1:.4f}")

# Detailed classification report
print(f"\n Transformer Detailed Performance:")
print(classification_report(y_val_labels, y_pred_labels, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_val_labels, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Transformer Confusion Matrix - Validation')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('transformer_confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
plt.show()


def evaluate_transformer_test(test_data, set_name):
    test_df = pd.DataFrame(test_data)
    X_test = test_df.drop(columns=['state'], axis=1)
    y_test = test_df['state']
    
    # Feature engineering
    X_test_eng = create_transformer_features(X_test)
    X_test_scaled = scaler.transform(X_test_eng)
    
    # Sequences
    y_test_encoded = le.transform(y_test)
    X_test_seq, y_test_seq = create_transformer_sequences(X_test_scaled, y_test_encoded, SEQUENCE_LENGTH)
    
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
    
    print(f"\n📊 Transformer {set_name}:")
    print(f"🎯 Accuracy: {acc:.4f}")
    print(f"📊 F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_labels, y_test_pred_labels, labels=le.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Transformer Confusion Matrix - {set_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'transformer_confusion_matrix_{set_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return acc, f1

# Test evaluations
test1_acc, test1_f1 = evaluate_transformer_test(test1_data, "Test Set 1")
test2_acc, test2_f1 = evaluate_transformer_test(test2_data, "Test Set 2")

# Performance comparison
comparison_data = {
    'Dataset': ['Validation', 'Test Set 1', 'Test Set 2'],
    'Accuracy': [val_acc, test1_acc, test2_acc],
    'F1-Score': [val_f1, test1_f1, test2_f1]
}

plt.figure(figsize=(10, 6))
x = np.arange(len(comparison_data['Dataset']))
width = 0.35
plt.bar(x - width/2, comparison_data['Accuracy'], width, label='Accuracy', alpha=0.8, color='gold')
plt.bar(x + width/2, comparison_data['F1-Score'], width, label='F1-Score', alpha=0.8, color='orange')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.title('Transformer Performance Comparison')
plt.xticks(x, comparison_data['Dataset'])
plt.legend()
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('transformer_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Final summary
print(f"\n🤖 TRANSFORMER FINAL SUMMARY")
print("="*45)
print(f"📊 Validation F1: {val_f1:.4f}")
print(f"📊 Test1 F1: {test1_f1:.4f}")
print(f"📊 Test2 F1: {test2_f1:.4f}")

all_scores = [val_f1, test1_f1, test2_f1]
mean_score = np.mean([test1_f1, test2_f1])
std_score = np.std(all_scores)

print(f"📊 Overall Mean F1: {mean_score:.4f}")
print(f"📊 Std Dev: {std_score:.4f}")

if mean_score >= 0.80:
    print(" OUTSTANDING!")
elif mean_score >= 0.75:
    print("EXCELLENT Transformer performance!")
elif mean_score >= 0.70:
    print("VERY GOOD! Transformer performance!")
else:
    print(" Good Transformer performance")

print(f"\n💾 Transformer Graphics Saved:")
print("  📊 transformer_class_distribution.png")
print("  🔗 transformer_feature_correlation.png")
print("  📈 transformer_train_distribution.png")
print("  📉 transformer_training_history.png")
print("  📋 transformer_confusion_matrix_validation.png")
print("  📋 transformer_confusion_matrix_test_set_1.png")
print("  📋 transformer_confusion_matrix_test_set_2.png")
print("  📊 transformer_performance_comparison.png")

print(f"\n Transformer motor fault detection completed!")
print(f" Transformer with {mean_score:.1%} Test F1-Score!")