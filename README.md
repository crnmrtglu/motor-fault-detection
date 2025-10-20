A comprehensive comparative analysis of machine learning algorithms for industrial motor fault detection systems, achieving 76.96% F1-score with LSTM-based approach.

🎯 Overview
This project presents a systematic evaluation of 6 different machine learning algorithms for detecting motor faults in industrial settings. The system classifies 6 motor conditions:

⚡ current_overload: Excessive current consumption
🔧 mechanical_bearing: Bearing failures
🌀 stator_winding: Stator winding faults
📉 insufficient_current: Insufficient current conditions
⚙️ rotor_bar: Rotor bar failures
✅ healthy: Normal operation state

✨ Key Features
Advanced Feature Engineering: Domain-specific features including current magnitude, vibration patterns, and efficiency proxies
Multiple Algorithm Comparison: LSTM, GRU, TCN, Transformer, Random Forest, XGBoost
Robust Evaluation: Cross-validation with multiple test datasets
Class Imbalance Handling: Weighted loss functions and sampling strategies
Temporal Pattern Recognition: Sequence-based analysis with 20-sample windows

📊 Dataset
⚠️ Note: The original dataset is not included in this repository due to confidentiality agreements.
Dataset Specifications

Size: 120,000+ motor sensor observations
Format: CSV files with 7 columns
Sensors:
Current Sensors: 3-phase measurements (current_R, current_S, current_W)
Vibration Sensors: 3-axis acceleration (X, Y, Z)


Class Distribution:

Healthy: 50.0% (~60,000 samples)
Stator winding: 14.8% (~18,000 samples)
Mechanical bearing: 10.3% (~12,400 samples)
Current overload: 9.7% (~11,600 samples)
Rotor bar: 7.8% (~9,400 samples)
Insufficient current: 7.4% (~8,900 samples)


📈 Results
LSTM Performance Highlights

Validation F1: 99.44%
Test Set 1 F1: 78.64%
Test Set 2 F1: 75.29%
Average F1: 76.96%

Best Performance Classes:

⚡ Stator Winding: Near-perfect detection (3110/3114 correct)
🔋 Current Overload: Strong performance (2888/2947 correct)
📉 Insufficient Current: Solid classification (2756/2956 correct)

Challenging Classes:

🔧 Mechanical Bearing: Most difficult (481/2939 correct - requires advanced vibration analysis)
⚙️ Rotor Bar: Moderate difficulty (1858/2846 correct - subtle symptoms)
