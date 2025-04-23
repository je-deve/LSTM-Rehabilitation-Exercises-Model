# LSTM Rehabilitation Exercise Model

This repository contains the end-to-end pipeline for extracting pose landmarks from videos of two different exercises (Heel Exercise and Chair Exercise), computing handcrafted features, training LSTM models to predict a “performance score".

---

## Initial Model Pipeline

1. **Raw Videos → Landmark Extraction**  
   - MediaPipe Pose detects 33 body landmarks per frame.  
   - We record normalized `(x,y,visibility)` for each landmark plus frame metadata.

2. **Handcrafted Feature Computation**  
   - **Knee angle**: Hip–Knee–Ankle angle via vector math.  
   - **Heel-to-Hip distance**: Euclidean distance normalized by frame.  
   - **Posture**: Neck/torso inclination angles → “Good”/“Bad.”  
   - **Repetition count**: Count transitions between “Pulled/Extended” and “Rest.”  

3. **Preprocessing**  
   - Drop frames with no visible side.  
   - Clean outliers, forward-fill missing values.  
   - One-hot encode categorical fields (visibility, posture, leg state).

4. **Sequence Preparation**  
   - Group by video, sort by frame index, pad/truncate to fixed length (e.g. 700 frames).  
   - Flatten sequences → 2D, apply `StandardScaler`, reshape back → 3D.

5. **LSTM Regression**  
   - Simple 1-layer LSTM (64 units) + Dense(32) + output(1).  
   - MSE loss, track MAE metric.  
   - 5-fold cross-validation with early stopping.

6. **Performance Scoring**  
   - Compute global “ideal” benchmarks (mean pulled/extended vs rest).  
   - Calculate per‐video error vs ideal → map to 0–100 score.  
   - Provide real and predicted comparison plots.


