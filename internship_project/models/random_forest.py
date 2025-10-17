import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# plot settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print(" Random Forest Motor Fault Detection")
print("="*40)

# Data loading
train_data = pd.read_csv('internship_project/data/motor_training_data.csv')
test1_data = pd.read_csv('internship_project/data/motor_testing_data.csv')
test2_data = pd.read_csv('internship_project/data/motor_testing_data_6.csv')

df = pd.DataFrame(train_data)
X = df.drop(columns=['state'], axis=1)
y = df['state']

print(f" Dataset: {df.shape}")

# Class distribution
class_counts = y.value_counts()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
class_counts.plot(kind='bar', color='skyblue')
plt.title('Original Class Distribution')
plt.xticks(rotation=45)
plt.ylabel('Sample Count')
plt.subplot(1, 2, 2)
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
plt.title('Class Proportions')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Engineering
def create_features(X):
    """fundamental features for motor fault detection"""
    X_new = X.copy()

    # current features
    X_new['current_magnitude'] = np.sqrt(X['current_R']**2 + X['current_S']**2 + X['current_W']**2)
    X_new['current_std'] = X[['current_R', 'current_S', 'current_W']].std(axis=1)
    X_new['current_mean'] = X[['current_R', 'current_S', 'current_W']].mean(axis=1)
    X_new['current_imbalance'] = X_new['current_std'] / (X_new['current_mean'] + 1e-8)

    # vibration features
    X_new['vibration_magnitude'] = np.sqrt(X['acceleration_X']**2 + X['acceleration_Y']**2 + X['acceleration_Z']**2)
    X_new['vibration_std'] = X[['acceleration_X', 'acceleration_Y', 'acceleration_Z']].std(axis=1)

    # Interaction
    X_new['current_vibration_ratio'] = X_new['current_magnitude'] / (X_new['vibration_magnitude'] + 1e-8)

    # Phase imbalance
    X_new['phase_imbalance'] = np.std([X['current_R'], X['current_S'], X['current_W']], axis=0)
    
    return X_new

print(" Feature Engineering...")
X_engineered = create_features(X)
print(f" Features: {X.shape[1]} → {X_engineered.shape[1]}")

# Feature correlation
plt.figure(figsize=(15, 12))
correlation_matrix = X_engineered.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# SMOTEENN
print(" SMOTEENN balancing...")
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train_scaled, y_train)

print("After SMOTEENN:")
for state, count in Counter(y_resampled).items():
    print(f"  {state}: {count:,}")

# Sampling comparison
resampled_counts = Counter(y_resampled)
original_counts = Counter(y_train)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(original_counts.keys(), original_counts.values(), color='lightcoral')
plt.title('Before Balancing (Train)')
plt.xticks(rotation=45)
plt.ylabel('Sample Count')
plt.subplot(1, 2, 2)
plt.bar(resampled_counts.keys(), resampled_counts.values(), color='lightgreen')
plt.title('After Balancing (SMOTEENN)')
plt.xticks(rotation=45)
plt.ylabel('Sample Count')
plt.tight_layout()
plt.savefig('sampling_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Model
print("🌲 Random Forest training ...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features=0.6,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# cross-validation with SMOTEENN
pipeline = ImbPipeline([
    ('sampling', SMOTEENN(random_state=42)),
    ('classifier', model)
])

cv_scores = cross_val_score(pipeline, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
print(f"CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

# Fit model
model.fit(X_resampled, y_resampled)

# Feature importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': X_engineered.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
plt.title('Top 15 Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Validation results
y_pred = model.predict(X_val_scaled)
val_acc = accuracy_score(y_val, y_pred)
val_f1 = f1_score(y_val, y_pred, average='weighted')

print(f"\n Validation Results:")
print(f" Accuracy: {val_acc:.4f}")
print(f" Weighted F1: {val_f1:.4f}")

# Confusion matrix for validation
cm = confusion_matrix(y_val, y_pred, labels=model.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_validation_set.png', dpi=300, bbox_inches='tight')
plt.show()

# Test sets
def test_model(test_data, set_name):
    test_df = pd.DataFrame(test_data)
    X_test = test_df.drop(columns=['state'], axis=1)
    y_test = test_df['state']
    
    X_test_eng = create_features(X_test)
    X_test_scaled = scaler.transform(X_test_eng)
    y_test_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\n {set_name}:")
    print(f" Accuracy: {acc:.4f}")
    print(f" F1-Score: {f1:.4f}")
    
    #  Confusion matrix for test set
    cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'Confusion Matrix - {set_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{set_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return acc, f1

test1_acc, test1_f1 = test_model(test1_data, "Test Set 1")
test2_acc, test2_f1 = test_model(test2_data, "Test Set 2")

# Performance comparison
scores_comparison = {
    'Dataset': ['Validation', 'Test Set 1', 'Test Set 2'],
    'Accuracy': [val_acc, test1_acc, test2_acc],
    'Weighted F1': [val_f1, test1_f1, test2_f1]
}
comparison_df = pd.DataFrame(scores_comparison)

plt.figure(figsize=(10, 6))
x = np.arange(len(comparison_df['Dataset']))
width = 0.35
plt.bar(x - width/2, comparison_df['Accuracy'], width, label='Accuracy', alpha=0.8)
plt.bar(x + width/2, comparison_df['Weighted F1'], width, label='Weighted F1', alpha=0.8)
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, comparison_df['Dataset'])
plt.legend()
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary
print(f"\n SUMMARY ")
print("="*30)
print(f"📊 CV F1: {cv_scores.mean():.4f}")
print(f"📊 Val F1: {val_f1:.4f}")
print(f"📊 Test1 F1: {test1_f1:.4f}")
print(f"📊 Test2 F1: {test2_f1:.4f}")

# consistency
all_scores = [cv_scores.mean(), val_f1, test1_f1, test2_f1]
mean_score = np.mean([test1_f1, test2_f1])
std_dev = np.std(all_scores)
print(f"Std Dev: {std_dev:.4f}")

if std_dev < 0.05:
    print(" Model is consistent!")
else:
    print(" Model is inconsistent!")

if mean_score > 0.80:
    print(" Model performance is good!")
elif mean_score > 0.60:
    print(" Model performance is average.")
else:
    print(" Model performance is poor.")

print("\n saved plots:")
print("   class_distribution.png")
print("   feature_correlation.png") 
print("   sampling_comparison.png")
print("   feature_importance.png")
print("   confusion_matrix_validation_set.png")
print("   confusion_matrix_test_set_1.png")
print("   confusion_matrix_test_set_2.png")
print("   performance_comparison.png")