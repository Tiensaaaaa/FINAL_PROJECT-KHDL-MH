# Predict Hit Song from Spotify - Depend on Artist 

This project focuses on building a machine learning model to predict whether a Spotify track is likely to become a **Hit** based on track, artist, album, genre, and release-related information.

## 👥 Thành viên nhóm & Phân công công việc

Dự án được thực hiện bởi nhóm [Tên Nhóm], với sự đóng góp cụ thể của từng thành viên như sau:

| Họ và Tên | Vai trò | Chi tiết công việc thực hiện |
| :--- | :--- | :--- |
| **Đỗ Minh Huyền** | Trưởng nhóm / Data Analysist | Finding data, lên ý tưởng, và viết baseline trong file `2.Data clean & preprocessing.ipynb` and write Model Tunning and Final Model in file `5.Model.ipynb` |
| **Trần Huệ Chi** | Data Analysist / Data Cleaner |Finding data, Viết file `3.EDA.ipynb`để trực quan hóa dữ liệu gốc và file `5.Model.ipynb`. |
| **Phạm Ngọc Yến Nhi** |Data Cleaner/ Data Analysist | Finding data, Write file `1.Importing library & loading data.ipynb`, file `4.Feature Engineering` và phần Pipline, data spliting trong file `5.Model.ipynb`  |
| **Nguyễn Huyền Trang** |Data Cleaner / Data Analysist |Finding data, Write file `2.Data clean & preprocessing.ipynb`, cleaning raw data, check logic, drop unnecessary columns and write file Readme |


## 1. Architecture

```text
Raw Spotify Track Data
        │
        ▼
Data Cleaning & Invalid Value Filtering
        │
        ▼
Feature Engineering
(duration_category / popularity_category / genre_group)
        │
        ▼
Time-based Train / Validation / Test Split
        │
        ▼
Preprocessing Transformer
(StandardScaler + OneHotEncoder + ColumnTransformer)
        │
        ▼
Baseline Models
(Dummy Classifier / Simple Models)
        │
        ▼
Machine Learning Models
(Random Forest / Tuned Random Forest with Optuna)
        │
        ▼
Evaluation
(F1-score / Precision / Recall / Accuracy / Confusion Matrix)
        │
        ▼
Final Model & Key Findings
(Tuned Random Forest for Hit Song Prediction)
```

## 2. Goal

The goal of this project is to build a machine learning model that predicts whether a Spotify track is likely to become a **Hit** based on artist, album, genre, release, and track-level features.

In this project, the target variable is defined as:

| Class | Meaning | Condition |
|---|---|---|
| `0` | Not Hit | `track_popularity <= 80` |
| `1` | Hit | `track_popularity > 80` |

Because the dataset is imbalanced, the main objective is not only to maximize accuracy, but also to improve the model’s ability to correctly identify the minority class, which is the **Hit** class.

Therefore, the project focuses more on **Recall**, **F1-score**, and **ROC-AUC** rather than accuracy alone.

---

## 3. Data

The project uses a Spotify track dataset containing information about tracks, artists, albums, genres, popularity, release dates, and duration.

After data cleaning and preprocessing, the final dataset contains:

| Item | Value |
|---|---:|
| Number of observations | 8,582 |
| Number of columns after cleaning | 17 |
| Target variable | `is_hit` |
| Prediction task | Binary classification |
| Main positive class | `Hit` |

### Main groups of variables

| Variable group | Examples | Purpose |
|---|---|---|
| Track information | `track_duration_min`, `track_number`, `explicit` | Describe basic characteristics of each song |
| Artist information | `artist_popularity`, `artist_followers`, `artist_genres` | Capture the influence and popularity of the artist |
| Album information | `album_type`, `album_total_tracks` | Describe album-level characteristics |
| Release information | `release_year`, `release_month`, `song_age`, `is_recent_song` | Capture time-related effects |
| Engineered features | `artist_power_score`, `artist_pop_x_followers`, `duration_x_artist_pop`, genre indicators | Improve model learning by adding interaction and transformed features |

### Data splitting strategy

The final model uses a **time-based train-validation-test split**. This is more realistic because the model learns from older songs and is tested on newer songs.

| Dataset | Year range | Number of rows | Hit rate |
|---|---:|---:|---:|
| Train | 1952–2021 | 6,178 | 0.063 |
| Validation | 2022–2023 | 991 | 0.057 |
| Test | 2024–2025 | 1,413 | 0.103 |

---

## 4. Evaluation Metrics

Since the dataset is imbalanced, accuracy alone can be misleading. A model may achieve high accuracy by predicting most songs as `Not Hit`, but this does not help identify actual hit songs.

The project uses the following evaluation metrics:

| Metric | Formula / Meaning | Why it is used |
|---|---|---|
| Accuracy | Correct predictions / Total predictions | Measures overall prediction correctness |
| Precision | TP / (TP + FP) | Shows how many predicted Hit songs are actually Hit |
| Recall | TP / (TP + FN) | Shows how many actual Hit songs the model can detect |
| F1-score | Harmonic mean of Precision and Recall | Balances Precision and Recall, especially useful for imbalanced data |
| ROC-AUC | Area under ROC curve | Measures the model’s ability to separate Hit and Not Hit classes |
| Log Loss | Penalizes wrong probability predictions | Evaluates probability quality |
| Brier Score | Mean squared error of predicted probabilities | Measures calibration of predicted probabilities |

For this project, **Hit Recall** and **Hit F1-score** are especially important because the main goal is to find potential hit songs.

---

## 5. Results

### Model comparison

| Model | Type | Accuracy | Hit Precision | Hit Recall | Hit F1-score | ROC-AUC | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | Baseline | 0.9313 | 0.0000 | 0.0000 | 0.0000 | N/A | High accuracy but failed to detect Hit tracks |
| Random Forest | Baseline | 0.9301 | 0.4600 | 0.0900 | 0.1500 | N/A | Detected some Hit tracks, but recall was still low |
| Tuned Random Forest | Main Model | 0.7721 | 0.2427 | 0.5685 | 0.3402 | 0.8009 | Best model for identifying Hit tracks |

### Final model metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7721 |
| Hit Precision | 0.2427 |
| Hit Recall | 0.5685 |
| Hit F1-score | 0.3402 |
| ROC-AUC | 0.8009 |
| Log Loss | 0.4558 |
| Brier Score | 0.1434 |

### Confusion matrix of final model

| Actual / Predicted | Not Hit | Hit |
|---|---:|---:|
| Not Hit | 1,008 | 259 |
| Hit | 63 | 83 |

The final model correctly detected **83 out of 146 actual Hit tracks** in the test set.

---

## 6. Key Findings

1. **Accuracy is not the best metric for this project.**  
   The Logistic Regression baseline achieved high accuracy of **93.13%**, but its Hit Recall and Hit F1-score were both **0.0000**. This means the model almost completely ignored the Hit class.

2. **The final tuned Random Forest model improves Hit detection significantly.**  
   Compared with the baseline Random Forest model:

   | Metric | Baseline Random Forest | Tuned Random Forest | Change |
   |---|---:|---:|---:|
   | Hit Recall | 0.0900 | 0.5685 | +0.4785 points / +531.67% |
   | Hit F1-score | 0.1500 | 0.3402 | +0.1902 points / +126.80% |
   | Accuracy | 0.9301 | 0.7721 | -0.1580 points / -16.99% |
   | Hit Precision | 0.4600 | 0.2427 | -0.2173 points / -47.24% |

3. **The trade-off is acceptable for the project objective.**  
   Although accuracy decreased from **0.9301** to **0.7721**, the model became much better at identifying actual Hit songs. This is more meaningful because the project focuses on detecting potential hits, not simply predicting the majority class.

4. **The model improved from detecting only 9% of Hit songs to detecting 56.85% of Hit songs.**  
   This shows that class balancing, feature engineering, time-based splitting, and hyperparameter tuning helped the model handle the imbalanced dataset more effectively.

5. **Artist-related features are the strongest predictors.**  
   The most important features are mainly related to artist popularity and audience size, such as `artist_followers_log`, `artist_is_popular`, `artist_popularity`, `artist_followers`, and `artist_pop_x_followers`.

6. **The final model is more useful in practice.**  
   A high-accuracy model that cannot detect Hit songs has limited value. The tuned Random Forest is more practical because it can identify more potential Hit tracks, even though it produces more false positives.

---

## 7. Conclusion

This project shows that predicting hit songs is challenging because Hit tracks make up a small proportion of the dataset. Baseline models achieved high accuracy but performed poorly on the minority class.

After applying feature engineering, time-based validation, class balancing, and Optuna hyperparameter tuning, the final Random Forest model achieved a much stronger ability to detect Hit tracks. The most important improvement is that **Hit Recall increased by 531.67%** and **Hit F1-score increased by 126.80%** compared with the baseline Random Forest model.

Therefore, the final model is considered more suitable for the project goal of identifying potential hit songs.
