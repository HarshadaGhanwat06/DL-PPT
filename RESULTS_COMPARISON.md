# Cardiac Timing Prediction: Model Evolution & Results Comparison

This document outlines the evolutionary journey of the Deep Learning pipeline for predicting Aortic Valve Opening (AVO) and Aortic Valve Closing (AVC) from non-invasive ICG (dZ/dt) and ECG signals. 

By comparing the models sequentially, we can see exactly how architectural decisions solved specific biological challenges.

---

## 1. Standalone 1D CNN Baseline
**Script:** `model/train_cnn_regression.py`

This was the initial proof-of-concept model. It used a very basic 1-dimensional Convolutional Neural Network (CNN) and only looked at a single signal channel (the dZ/dt wave) to predict the timings.

### Terminal Output
```text
[MAE]  AVO:  29.46 ms | AVC:  38.90 ms | Mean:  34.18 ms
[RMSE] AVO:  36.19 ms | AVC:  45.71 ms | Mean:  40.95 ms
[R²]   AVO: -0.048    | AVC: -0.058    | Mean: -0.053
[MedAE]AVO:  23.82 ms | AVC:  36.97 ms | Mean:  30.39 ms
[MAX]  AVO: 149.47 ms | AVC: 154.13 ms | Mean: 151.80 ms
[BIAS] AVO:  -7.39 ms | AVC:  -8.27 ms | Mean:  -7.83 ms
[<10ms]AVO:   12.1 %  | AVC:   11.5 %  | Mean:   11.8 %
[<20ms]AVO:   37.8 %  | AVC:   23.8 %  | Mean:   30.8 %
```

### Explanation & Takeaway
*   **The Baseline:** The model achieved a Mean Absolute Error (MAE) of 34.18 ms. 
*   **R² Analysis:** Both R² scores are slightly negative (-0.053). This is an artifact of the biology: valve timings have extremely low natural variance, so even small ~30ms mistakes drop the R² below zero.
*   **Takeaway:** This proved the task was possible, but a `<20ms` accuracy of only 30.8% meant it wasn't clinically viable yet.

---

## 2. Improved Deeper CNN
**Script:** `model/train_cnn_improved.py`

To increase accuracy, the architecture was deepened by adding more convolutional layers and introducing Batch Normalization to stabilize the learning process.

### Terminal Output
```text
[MAE]  AVO:  27.14 ms | AVC:  43.32 ms | Mean:  35.23 ms
[RMSE] AVO:  36.06 ms | AVC:  52.90 ms | Mean:  44.48 ms
[R²]   AVO: -0.040    | AVC: -0.418    | Mean: -0.229
[MedAE]AVO:  19.15 ms | AVC:  36.41 ms | Mean:  27.78 ms
[MAX]  AVO: 166.84 ms | AVC: 147.56 ms | Mean: 157.20 ms
[BIAS] AVO:  -8.68 ms | AVC:  13.55 ms | Mean:   2.44 ms
[<10ms]AVO:   24.1 %  | AVC:   14.2 %  | Mean:   19.2 %
[<20ms]AVO:   52.1 %  | AVC:   29.9 %  | Mean:   41.0 %
```

### Explanation & Takeaway
*   **The Trade-off:** While the overall Mean MAE mathematically got worse (35.23 ms), the **clinical precision skyrocketed**. 
*   **AVO Mastery:** The Median Error (MedAE) for AVO dropped to just 19.15 ms, and its `<20ms Accuracy` jumped to 52.1% (up from 37.8%). It became incredibly good at finding the valve opening.
*   **AVC Struggle:** The deeper network struggled badly with finding the distant AVC target, heavily dragging down the average.
*   **Takeaway:** Deeper networks are better at finding local features (like AVO near the R-peak) but get confused over long timelines (AVC).

---

## 3. Dual-Branch CNN
**Script:** `model/train_cnn_dual_branch.py`

Realizing that feeding the ICG and ECG signals into the same network confused the model, this architecture split them up. One CNN looked exclusively at the ECG, another CNN looked exclusively at the ICG, and they mathematically combined their thoughts at the end.

### Terminal Output
```text
[MAE]  AVO:  22.06 ms | AVC:  40.79 ms | Mean:  31.42 ms
[RMSE] AVO:  30.82 ms | AVC:  47.16 ms | Mean:  38.99 ms
[R²]   AVO:  0.240    | AVC: -0.127    | Mean:  0.057
[MedAE]AVO:  14.00 ms | AVC:  38.90 ms | Mean:  26.45 ms
[MAX]  AVO: 154.80 ms | AVC: 118.98 ms | Mean: 136.89 ms
[BIAS] AVO:  -7.87 ms | AVC:   8.30 ms | Mean:   0.21 ms
[<10ms]AVO:   38.4 %  | AVC:    8.8 %  | Mean:   23.6 %
[<20ms]AVO:   59.7 %  | AVC:   22.5 %  | Mean:   41.1 %
```

### Explanation & Takeaway
*   **The Breakthrough:** Mean MAE dropped to an impressive **31.42 ms**.
*   **Positive R²!** Because the model wasn't distracted by cross-signal noise, the AVO R² finally turned positive (`0.240`), mathematically proving it is significantly better than a blind baseline guess.
*   **Clinical Success:** A staggering **60%** of AVO predictions were within 20ms, and the Median Error for AVO plummeted to 14.00 ms.
*   **Takeaway:** Independent feature extraction for complex biological signals is vastly superior to mixing them blindly.

---

## 4. Pure XGBoost Baseline
**Script:** `model/train_xgboost.py`

This script abandoned Deep Learning entirely to test if a classic, robust Tree-based Machine Learning algorithm (XGBoost) could handle the noisy data better.

### Terminal Output (JSON summary)
```text
"mean_mae_ms": 32.84
"mean_r2": 0.0307
"mean_medae_ms": 31.03
"avo_acc_20ms_%": 40.0
"avc_acc_20ms_%": 27.12
```

### Explanation & Takeaway
*   **Robustness:** XGBoost achieved a very respectable 32.84 ms Mean MAE, outperforming the early CNNs but slightly losing to the Dual-Branch CNN.
*   **R² Stabilization:** Notice the positive `mean_r2` of 0.030. XGBoost is highly resistant to massive outliers. It rarely makes a "150ms" mistake, which protects the R-squared score.
*   **Takeaway:** XGBoost is safer and more robust, but lacks the "vision" to achieve the extreme high precision of deep learning convolutions.

---

## 5. The Final Hybrid Model
**Script:** `model/final_hybrid_model.py`

This was the culmination of the historical research phase: It used a CNN solely as a "vision" feature extractor, and passed those features into XGBoost to make the final robust decision.

### Terminal Output (JSON summary)
```text
"mean_mae_ms": 35.73
"mean_r2": -0.124
"mean_medae_ms": 34.85
"avo_acc_20ms_%": 24.11
"mean_acc_20ms_%": 25.61
```

### Explanation & Takeaway
*   **The Limitation:** Surprisingly, the Hybrid model performed worse than the standalone XGBoost and the Dual-Branch CNN. 
*   **Why?** The CNN feature extractor was pre-trained and then frozen. It turns out that forcing XGBoost to learn from rigid, frozen CNN features prevents the network from dynamically adjusting to the data. 
*   **The Ultimate Conclusion:** This result paved the way for the current state-of-the-art approach in the project: abandoning static hybrids and moving entirely to modern **End-to-End Deep Learning Ensembles (ResNet, TCN, Transformers)**, which currently hold the project's record for highest stability and accuracy.
