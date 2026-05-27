# 📝 Sentiment Analysis on McDonald’s Store Reviews

This project analyzes customer reviews of McDonald’s locations across the United States using sentiment analysis. The goal is to predict customer sentiment from review text, offering insights that can help McDonald’s enhance customer satisfaction and service quality.

---

## 📌 Project Overview

Using a combination of data collected from Google Maps API and a Kaggle dataset, this project classifies customer reviews as **positive** or **negative** based on their content. By leveraging NLP techniques and various machine learning models, we identify which approach performs best for binary sentiment classification.

---

## 📂 Dataset Sources

1. **Google Maps API:**  
   - Collected 2,800+ McDonald's locations across all 50 U.S. states  
   - Extracted over 20,000 reviews including text, star rating, and geolocation

2. **Kaggle McDonald’s Reviews Dataset:**  
   - Provided additional reviews and store information  
   - Manually corrected and cleaned some missing entries

---

## 🧹 Data Preprocessing

- Cleaned review text (removed special characters, punctuation, newlines, and short words)
- Merged datasets and standardized columns
- Converted star ratings into binary sentiment:
  - Ratings 1–2 → **Negative**
  - Ratings 3–5 → **Positive**

---

## 🧠 Modeling

**Text Vectorization Techniques Used:**
- TF-IDF
- Word2Vec
- GloVe Embeddings

**Machine Learning Models Evaluated:**
- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest
- Naive Bayes
- Gradient Boosting

---

## ✅ Best Model

**Model:** Support Vector Classifier (SVC)  
**Vectorization:** TF-IDF  
**Test Accuracy:** **0.8912**

TF-IDF + SVC consistently outperformed more complex embeddings like Word2Vec and GloVe, indicating its effectiveness on short, informal text like customer reviews.

---

## 📊 Model Comparison

| Model                     | Accuracy |
|--------------------------|----------|
| Logistic Regression      | 0.8834   |
| Support Vector Classifier| **0.8909** |
| Random Forest            | 0.8752   |
| Naive Bayes              | 0.8752   |
| Gradient Boosting        | 0.8280   |

**SVC (Test Accuracy):** 0.8912  
**Word2Vec + SVC:** 0.8790  
**GloVe + SVC:** 0.8269  

---

## 📈 Visualizations

- Distribution of positive vs. negative reviews
- Word clouds for each sentiment class
- Most frequent positive/negative keywords

---

## 🔧 Tools & Libraries

- Python, Pandas, NumPy
- scikit-learn, XGBoost
- NLTK, Gensim
- Seaborn, Matplotlib
- Google Maps API

---

## ✍️ Author

**Benny Chen**  
Graduate Student, M.S. in Data Science  
Maryville University 

---

## 📌 License

This project is for educational purposes only and not affiliated with McDonald's Corporation.
