# Predict Hit Song from Spotify - Depend on Artist 

This project focuses on building a machine learning model to predict whether a Spotify track is likely to become a **Hit** based on track, artist, album, genre, and release-related information. This project can be applied in the music streaming and entertainment industry to support song success prediction, recommendation systems, and marketing strategies. By identifying tracks with high potential to become “Hit” songs, music platforms and producers can make more data-driven decisions regarding promotion, playlist recommendations, and audience targeting
## 👥 Team Member & Task Allocation

The project was carried out by Group 1, with each member contributing to specific tasks as follows:

| Full Name | Role | Specific role |
| :--- | :--- | :--- |
| **Đỗ Minh Huyền** | Leader / Data Analysist | Finding data, Write file `3.Baseline_Model.ipynb`, Write Model Tunning and Final Model in file `5.Model.ipynb`, edit Readme, Format the notebook. |
| **Trần Huệ Chi** | Data Analysist / Data Cleaner |Finding data, Write file `2.EDA.ipynb` to visualize the raw data and Write Feature importance and Saving Model in file `5.Model.ipynb`, Edit Readme. |
| **Phạm Ngọc Yến Nhi** |Data Cleaner/ Data Analysist | Finding data, Write file `4.Feature_Engineering` and Pipline, Data spliting parts in file `5.Model.ipynb`, Format the notebook.  |
| **Nguyễn Huyền Trang** |Data Cleaner / Data Analysist |Finding data, Write file `1.Data_Cleaning_and_Preprocessing.ipynb`, cleaning raw data, check logic, drop unnecessary columns and Edit file Readme. |

## 1. Data

The project uses a Spotify track dataset containing information about tracks, artists, albums, genres, popularity, release dates, and duration.
- link data: (https://www.kaggle.com/datasets/alyahmedts13/spotify-songs-for-ml-and-analysis-over-8700-tracks?select=spotify_data+clean.csv)

After data cleaning and preprocessing, the final dataset contains:

| Item | Value |
|---|---:|
| Number of observations | 8,582 |
| Number of columns after cleaning | 17 |
| Target variable | `is_hit` |
| Prediction task | Binary classification |
| Main positive class | `Hit` |

### 1.1. Main groups of variables

| Variable group | Examples | Purpose |
|---|---|---|
| Track information | `track_duration_min`, `track_number`, `explicit` | Describe basic characteristics of each song |
| Artist information | `artist_popularity`, `artist_followers`, `artist_genres` | Capture the influence and popularity of the artist |
| Album information | `album_type`, `album_total_tracks` | Describe album-level characteristics |
| Release information | `release_year`, `release_month`, `song_age`, `is_recent_song` | Capture time-related effects |
| Engineered features | `artist_power_score`, `artist_pop_x_followers`, `duration_x_artist_pop`, genre indicators | Improve model learning by adding interaction and transformed features |


### 1.2. Metrics Strategy

Our dataset is **highly imbalanced**, with only about **7% Hit tracks** and **93% Not Hit tracks**. Therefore, **Accuracy can be misleading** because a model can achieve high accuracy by predicting most songs as Not Hit, while failing to detect actual Hit songs.

Thus, we employ evaluation metrics that focus more on the minority class `is_hit = 1`, including **Precision**, **Recall**, and **F1-score**.

- **Precision** shows how many tracks predicted as Hit are actually Hit.
- **Recall** shows how many actual Hit tracks the model can correctly detect.
- **F1-score** balances Precision and Recall, so it is more suitable for this imbalanced dataset.

In this project, **Hit F1-score is selected as the main metric**, while **Hit Recall is also considered important**, because the main goal is to improve the model’s ability to identify potential Hit tracks.

### 1.3. Data Splitting Strategy

The final model uses a **time-based train-validation-test split** because the dataset contains time-related information such as `release_year`. This helps avoid data leakage, since the model learns from older songs and is tested on newer songs.

- **Training set**: older songs for model training.
- **Validation set**: middle-period songs for tuning and comparison.
- **Test set**: newer songs for final evaluation.

A fully random split is not preferred for the final model because this project has a time-series-like structure. Since the dataset is highly imbalanced, the class distribution of `is_hit` is checked carefully. In random split experiments, `stratify=y` is used to keep the Hit/Not Hit ratio consistent.

| Dataset | Year range | Number of rows | Hit rate |
|---|---:|---:|---:|
| Train | 1952–2021 | 6,178 | 0.063 |
| Validation | 2022–2023 | 991 | 0.057 |
| Test | 2024–2025 | 1,413 | 0.103 |

---

## 2. Architecture

```text
Raw Spotify Track Data
        │
        ▼
Data Cleaning & Invalid Value Filtering
(remove duplicate rows, handle missing values, fix data types, 
convert release date, remove invalid popularity, duration, followers, 
and album track values)
        │
        ▼
Exploratory Data Analysis (EDA)
(analyze data distribution, class imbalance, relationships between features, 
and music-related patterns before building models)
        │
        ▼
Baseline Models
(Random Forest, Logistic Regression)
        │
        ▼
Feature Engineering 
(create new features such as duration_category, popularity_category, 
artist_followers_category, genre_group, release_year, album_type, 
and album_total_tracks to help the model capture music-related patterns)
        │
        ▼
Time-based Train / Validation / Test Split
        │
        ▼
Preprocessing Transformer
(StandardScaler + OneHotEncoder + ColumnTransformer)
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

## 3. Goal

The goal of this project is to build a machine learning model that predicts whether a Spotify track is likely to become a **Hit** based on artist, album, genre, release, and track-level features.

In this project, the target variable is defined as:

| Class | Meaning | Condition |  Target Distribution |
|---|---|---|---|
| `0` | Not Hit | `track_popularity <= 80` | 7,990 (~93.1%) |
| `1` | Hit | `track_popularity > 80` | 592 (~6.9%) |

Because the dataset is imbalanced, the main objective is not only to maximize accuracy, but also to improve the model’s ability to correctly identify the minority class, which is the **Hit** class.

Therefore, the project focuses more on **Recall**, **F1-score**, and **ROC-AUC** rather than accuracy alone.

---



## 4. Evaluation Metrics

Since the dataset is imbalanced, accuracy alone can be misleading. A model may achieve high accuracy by predicting most songs as `Not Hit`, but this does not help identify actual hit songs.

The project uses the following evaluation metrics:

| Metric | Formula / Meaning | Why it is used |
|---|---|---|
| Accuracy | Correct predictions / Total predictions | Measures overall prediction correctness |
| Precision | TP / (TP + FP) | Shows how many predicted Hit songs are actually Hit |
| Recall | TP / (TP + FN) | Shows how many actual Hit songs the model can detect |
| **F1-score (priority/ main metric)** | Harmonic mean of Precision and Recall | Balances Precision and Recall, especially useful for imbalanced data |
| ROC-AUC | Area under ROC curve | Measures the model’s ability to separate Hit and Not Hit classes |

For this project, **Hit Recall** and **Hit F1-score** are especially important because the main goal is to find potential hit songs.

---

## 5. Results

### Model comparison

| Model | Type | Accuracy | Hit Precision | Hit Recall | Hit F1-score | Notes |
|---|---|---:|---:|---:|---:|---|
| Logistic Regression | Baseline | 0.9313 | 0.0000 | 0.0000 | 0.0000 | High accuracy but failed to detect Hit tracks |
| Random Forest | Baseline | 0.9301 | 0.4600 | 0.0900 | 0.1500 | Detected some Hit tracks, but recall was still low |
| **Tuned Random Forest** | Main Model | 0.7693 | 0.2353 | 0.5479 | 0.3292 | Best model for identifying Hit tracks |

The **Tuned Random Forest** achieved the best performance for detecting Hit tracks. Although its accuracy decreased to **0.7693**, this is acceptable because accuracy is not the main priority in this highly imbalanced dataset.

Compared with the baseline models, the tuned model improved the ability to identify Hit songs much better. Its **Hit Recall increased to 0.5479**, meaning the model can detect more than half of the actual Hit tracks. Its **Hit F1-score reached 0.3292**, which is the highest among all models.

This result shows that the tuned model is more suitable for this project because the main goal is not only to get high accuracy, but to improve prediction for the minority class `is_hit = 1`.

### Final model metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7693 |
| Hit Precision | 0.2353 |
| Hit Recall | 0.5479 |
| Hit F1-score | 0.3292 |
| ROC-AUC | 0.8003 |
| Log Loss | 0.4323 |
| Brier Score | 0.1368 |

### Confusion matrix of final model

| Actual / Predicted | Not Hit | Hit |
|---|---:|---:|
| Not Hit | 1,007 | 260 |
| Hit | 66 | 80 |

The final model correctly detected **80 out of 146 actual Hit tracks** in the test set.

---

## 6. Key Findings

1. **Accuracy is not the best metric for this project.**  
   The Logistic Regression baseline achieved a high accuracy of **93.13%**, but its Hit Recall and Hit F1-score were both **0.0000**. This means the model almost completely ignored the Hit class and mainly predicted tracks as Not Hit.

2. **The final Tuned Random Forest model improves Hit detection significantly.**  
   Compared with the baseline Random Forest model:

   | Metric | Baseline Random Forest | **Tuned Random Forest** | Change |
   |---|---:|---:|---:|
   | Hit Recall | 0.0900 | 0.5479 | **+0.4579 points / +508.78%** |
   | Hit F1-score | 0.1500 | 0.3292 | **+0.1792 points / +119.47%** |
   | Accuracy | 0.9301 | 0.7693 | -0.1608 points / -17.29% |
   | Hit Precision | 0.4600 | 0.2353 | -0.2247 points / -48.85% |

3. **The trade-off is acceptable for the project objective.**  
   Although accuracy decreased from **0.9301** to **0.7693**, the model became much better at identifying actual Hit songs. This is more meaningful because the main goal of the project is to detect potential hits, not simply predict the majority class.

4. **The model improved from detecting only 9% of Hit songs to detecting 54.79% of Hit songs.**  
   The baseline Random Forest detected only **9%** of actual Hit songs, while the Tuned Random Forest detected **54.79%**. This shows that feature engineering, class imbalance handling, time-based splitting, and hyperparameter tuning helped the model perform better on the minority class `is_hit = 1`.

5. **Artist-related features are the strongest predictors.**  
   Important features are mainly related to artist popularity and audience size, such as `artist_followers_log`, `artist_is_popular`, `artist_popularity`, `artist_followers`, and `artist_pop_x_followers`.  
   This suggests that songs by **popular artists** or **artists with many followers** are more likely to become Hits because they already have a large fanbase and higher initial audience reach.

6. **Album-related features also provide useful signals.**  
   Features such as `album_type` and `album_total_tracks` help explain how the release format affects hit potential.  
   For example, tracks released as singles may have a higher chance of becoming Hits because they are usually promoted more directly. In contrast, songs from albums or compilations may receive different levels of attention depending on the album structure and number of tracks.

7. **Genre helps capture differences in music popularity patterns.**  
   The `genre_group` feature helps simplify raw genre information into broader groups such as Pop, Hip Hop/Rap, Rock, Electronic, R&B, Indie, Latin, Jazz, and Other.  
   Some genres are more likely to appear among popular tracks because they have wider audiences and stronger streaming demand. Therefore, genre is an important feature for understanding differences in Hit probability.

8. **The final model is more useful in practice.**  
   A high-accuracy model that cannot detect Hit songs has limited practical value. The Tuned Random Forest is more useful because it can identify more potential Hit tracks, even though it produces more false positives.
   
## 7. Conclusion

This project shows that predicting hit songs is challenging because Hit tracks make up a small proportion of the dataset. Baseline models achieved high accuracy but performed poorly on the minority class.

After applying feature engineering, time-based validation, class balancing, and Optuna hyperparameter tuning, the final Random Forest model achieved a much stronger ability to detect Hit tracks. The most important improvement is that **Hit Recall increased by 508.78%** and **Hit F1-score increased by 119.47%** compared with the baseline Random Forest model.

Therefore, the final model is considered more suitable for the project goal of identifying potential hit songs.
