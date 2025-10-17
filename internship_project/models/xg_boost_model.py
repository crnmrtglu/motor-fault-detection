import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# XGBoost import
try:
    import xgboost as xgb
    print(" XGBoost yüklendi!")
except ImportError:
    print(" XGBoost yüklü değil. 'pip install xgboost' yapın!")
    import sys
    sys.exit(1)

# graph settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print(" XGBoost Motor Fault Detection")
print("="*40)

# data loading
train_data = pd.read_csv('internship_project/data/motor_training_data.csv')
test1_data = pd.read_csv('internship_project/data/motor_testing_data.csv')
test2_data = pd.read_csv('internship_project/data/motor_testing_data_6.csv')

df = pd.DataFrame(train_data)
X = df.drop(columns=['state'], axis=1)
y = df['state']

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f" Dataset: {df.shape}")

# CLASS DISTRIBUTION
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
plt.savefig('xgb_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Engineering
def create_features(X):
    """fundamental features for motor fault detection"""
    X_new = X.copy()

    # Current features
    X_new['current_magnitude'] = np.sqrt(X['current_R']**2 + X['current_S']**2 + X['current_W']**2)
    X_new['current_std'] = X[['current_R', 'current_S', 'current_W']].std(axis=1)
    X_new['current_mean'] = X[['current_R', 'current_S', 'current_W']].mean(axis=1)
    X_new['current_imbalance'] = X_new['current_std'] / (X_new['current_mean'] + 1e-8)

    # vibration features
    X_new['vibration_magnitude'] = np.sqrt(X['acceleration_X']**2 + X['acceleration_Y']**2 + X['acceleration_Z']**2)
    X_new['vibration_std'] = X[['acceleration_X', 'acceleration_Y', 'acceleration_Z']].std(axis=1)

    # Interaction
    X_new['current_vibration_ratio'] = X_new['current_magnitude'] / (X_new['vibration_magnitude'] + 1e-8)

    # phase imbalance
    X_new['phase_imbalance'] = np.std([X['current_R'], X['current_S'], X['current_W']], axis=0)
    
    return X_new

print(" Feature Engineering...")
X_engineered = create_features(X)
print(f" Features: {X.shape[1]} → {X_engineered.shape[1]}")

#  Feature correlation
plt.figure(figsize=(15, 12))
correlation_matrix = X_engineered.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('xgb_feature_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scaling

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Manual SMOTE strategy  - protect healthy
print(" Manual SMOTE balancing..")

# Show original distribution
print("Original training distribution:")
original_counts = Counter(y_train)
for encoded_label, count in original_counts.items():
    original_label = le.inverse_transform([encoded_label])[0]
    print(f"  {original_label}: {count:,}")

# protect healthy
healthy_encoded = le.transform(['healthy'])[0]
insufficient_encoded = le.transform(['insufficient_current'])[0] 
rotor_encoded = le.transform(['rotor_bar'])[0]
current_encoded = le.transform(['current_overload'])[0]
mechanical_encoded = le.transform(['mechanical_bearing'])[0]
stator_encoded = le.transform(['stator_winding'])[0]

# Manual sampling strategy
sampling_strategy = {
    insufficient_encoded: 15000,  # 5,717 → 15,000
    rotor_encoded: 15000,         # 5,995 → 15,000  
    current_encoded: 18000,       # 7,422 → 18,000
    mechanical_encoded: 18000,    # 7,916 → 18,000
    stator_encoded: 20000,        # 11,339 → 20,000
    # healthy: 38,391 
}

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

print("\n after smote:")
resampled_counts = Counter(y_resampled)
for encoded_label, count in resampled_counts.items():
    original_label = le.inverse_transform([encoded_label])[0]
    print(f"  {original_label}: {count:,}")

print(f"\n Healthy is protected: {original_counts[healthy_encoded]:,} → {resampled_counts[healthy_encoded]:,}")
print(f" Total: {len(y_train):,} → {len(y_resampled):,} ({len(y_resampled)/len(y_train):.1f}x)")

# Sampling comparison
resampled_counts = Counter(y_resampled)
original_counts = Counter(y_train)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
original_labels = [le.inverse_transform([x])[0] for x in original_counts.keys()]
plt.bar(original_labels, original_counts.values(), color='lightcoral')
plt.title('Before Balancing (Train)')
plt.xticks(rotation=45)
plt.ylabel('Sample Count')
plt.subplot(1, 2, 2)
resampled_labels = [le.inverse_transform([x])[0] for x in resampled_counts.keys()]
plt.bar(resampled_labels, resampled_counts.values(), color='lightgreen')
plt.title('After SMOTE')
plt.xticks(rotation=45)
plt.ylabel('Sample Count')
plt.tight_layout()
plt.savefig('xgb_sampling_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# XGBoost Model Hyperparameter Tuning
print(" XGBoost + Minimal Tuning...")


from imblearn.pipeline import Pipeline as ImbPipeline

# Base XGBoost
base_xgb = xgb.XGBClassifier(
    objective='multi:softprob',
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
    use_label_encoder=False
)

# minimal hyperparameter grid - just 4 combinations
param_grid = {
    'classifier__n_estimators': [200, 300],
    'classifier__max_depth': [6, 8]
}

# Basic Pipeline
pipeline = ImbPipeline([
    ('sampling', smote),
    ('classifier', base_xgb)
])

# Grid Search
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

print(" Minimal Grid Search (4 combinations)...")
grid_search.fit(X_train_scaled, y_train)

print(f"\n Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f" Best CV Score: {grid_search.best_score_:.4f}")

# Get the best model
best_model = grid_search.best_estimator_

# CV scores
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
print(f"📊 CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

# Fit final model
best_model.fit(X_train_scaled, y_train)

# Feature importance plot
feature_importance = best_model.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    'feature': X_engineered.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
plt.title('Top 15 XGBoost Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Validation results
y_pred = best_model.predict(X_val_scaled)
y_val_labels = le.inverse_transform(y_val)
y_pred_labels = le.inverse_transform(y_pred)

val_acc = accuracy_score(y_val_labels, y_pred_labels)
val_f1 = f1_score(y_val_labels, y_pred_labels, average='weighted')

print(f"\n Validation results:")
print(f" Accuracy: {val_acc:.4f}")
print(f"Weighted F1: {val_f1:.4f}")

# confusion matrix for validation 
cm = confusion_matrix(y_val_labels, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('xgb_confusion_matrix_validation_set.png', dpi=300, bbox_inches='tight')
plt.show()

# Test sets
def test_model(test_data, set_name):
    test_df = pd.DataFrame(test_data)
    X_test = test_df.drop(columns=['state'], axis=1)
    y_test = test_df['state']
    
    X_test_eng = create_features(X_test)
    X_test_scaled = scaler.transform(X_test_eng)
    y_test_encoded = le.transform(y_test)
    y_test_pred = best_model.predict(X_test_scaled)
    
    y_test_pred_labels = le.inverse_transform(y_test_pred)
    
    acc = accuracy_score(y_test, y_test_pred_labels)
    f1 = f1_score(y_test, y_test_pred_labels, average='weighted')
    
    print(f"\n📊 {set_name}:")
    print(f" accuracy: {acc:.4f}")
    print(f" F1-Score: {f1:.4f}")
    
    # Confusion matrix for test set
    cm = confusion_matrix(y_test, y_test_pred_labels, labels=le.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {set_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'xgb_confusion_matrix_{set_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
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
plt.title('XGBoost Model Performance Comparison')
plt.xticks(x, comparison_df['Dataset'])
plt.legend()
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('xgb_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
 # Training history visualization
eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
model_history = best_model.named_steps['classifier'].fit(
        X_train_scaled, y_train,
        eval_set=eval_set,
        verbose=False)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(model_history.evals_result()['validation_0']['mlogloss'], label='Train')
plt.plot(model_history.evals_result()['validation_1']['mlogloss'], label='Validation')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.title('XGBoost Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('xgb_learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary
print(f"\n XGBoost summary")
print("="*35)
print(f"📊 Best CV Score: {grid_search.best_score_:.4f}")
print(f"📊 CV F1: {cv_scores.mean():.4f}")
print(f"📊 Val F1: {val_f1:.4f}")
print(f"📊 Test1 F1: {test1_f1:.4f}")
print(f"📊 Test2 F1: {test2_f1:.4f}")

# Consistency
all_scores = [cv_scores.mean(), val_f1, test1_f1, test2_f1]
std_dev = np.std(all_scores)
mean_score = np.mean([test1_f1, test2_f1])
print(f" Average F1: {mean_score:.4f}")
print(f" Std Dev: {std_dev:.4f}")


if mean_score > 0.6:
    print(" XGBoost is above 60%!")
elif mean_score > 0.5:
    print(" XGBoost is performing well!")
else:
    print(" XGBoost needs improvement.")
    
print("\n💡 XGBoost motor fault detection ready!")
print("\n💾 Saved XGBoost graphics:")
print("  📊 xgb_class_distribution.png")
print("  🔗 xgb_feature_correlation.png") 
print("  ⚖️ xgb_sampling_comparison.png")
print("  📈 xgb_feature_importance.png")
print("  📋 xgb_confusion_matrix_validation_set.png")
print("  📋 xgb_confusion_matrix_test_set_1.png")
print("  📋 xgb_confusion_matrix_test_set_2.png")
print("  📊 xgb_performance_comparison.png")