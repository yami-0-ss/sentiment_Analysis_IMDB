
# 💬 Sentiment Analysis on IMDB Movie Reviews

A machine learning web app that tells you whether a movie review is **Positive** or **Negative**, shows how confident the model is, and explains **which words** led to the decision.

🔗 **Live demo:** [Open the app](https://sentimentanalysisimdb-das5panfpvvly7jkqkk5uu.streamlit.app/)

> If the app shows a "this app has gone to sleep" screen, click **Yes, get this app back up** and wait a few seconds. Free Streamlit apps sleep after a period of inactivity.

---

## 📌 Project Overview

| Item | Details |
|------|---------|
| **Problem** | Binary text classification (positive / negative sentiment) |
| **Dataset** | IMDB Dataset of 50K Movie Reviews (25,000 positive + 25,000 negative) |
| **Text features** | TF-IDF with unigrams and bigrams (single words and word pairs) |
| **Model** | Logistic Regression (compared with Naive Bayes and Linear SVM) |
| **Accuracy** | About 89-90% on the held-out test set |
| **Model training** | Google Colab |
| **Deployment** | Streamlit Community Cloud (via GitHub) |

---

## ✨ App Features

- **Single review analysis** with sentiment label, confidence score and a gauge chart
- **Explainability:** colored word chips and a bar chart show which words pushed the prediction towards positive or negative
- **Batch analysis:** upload a CSV file, analyze many reviews at once and download the results
- **History tab:** donut chart and score-over-time chart of your analyses
- **Theme switcher:** six color themes (Ocean, Sunset, Forest, Royal Purple, Dark Night, Clean Light) plus a custom accent color picker
- **Example reviews** to try the app quickly

---

## 🧠 How It Works

1. **Text cleaning:** lowercase the text and remove HTML tags, URLs, digits and punctuation. Stop words are kept on purpose because words like "not" change the meaning.
2. **TF-IDF vectorization:** converts text into numbers. Rare but meaningful words get higher weight than very common ones, and word pairs such as "not good" are captured.
3. **Logistic Regression:** predicts the probability that a review is positive.
4. **Explanation:** each word has a learned weight, so the app can show the words that contributed most to the result.

```
Raw review -> Clean text -> TF-IDF (1-2 grams) -> Logistic Regression -> Positive / Negative + confidence
```

---

## 📊 Model Comparison

Three models were trained on the same TF-IDF features, and the best one was selected by test accuracy.

| Model | Notes |
|-------|-------|
| Naive Bayes | Fast baseline |
| Linear SVM | Strong for text, but gives no probability output |
| **Logistic Regression** | Best balance of accuracy, probabilities and interpretability (used in the app) |

---

## 🔄 Project Workflow

1. Downloaded the IMDB 50K reviews dataset from Kaggle
2. Cleaned the text and split it into 80% training and 20% testing data
3. Trained and compared three models in **Google Colab**
4. Evaluated the best model with a classification report and confusion matrix
5. Saved the full pipeline (TF-IDF + model) as `sentiment_model.pkl` using joblib
6. Built the web app in `app.py` using Streamlit
7. Pushed the files to GitHub and deployed on **Streamlit Community Cloud**

---

## 📁 Project Structure

```
├── app.py                  # Streamlit web app
├── sentiment_model.pkl     # Trained model (TF-IDF + Logistic Regression pipeline)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ☁️ Deployment

The app is deployed on **Streamlit Community Cloud**:

1. Uploaded `app.py`, `sentiment_model.pkl`, `requirements.txt` and `README.md` to a GitHub repository
2. Signed in at [share.streamlit.io](https://share.streamlit.io) with GitHub
3. Selected the repository and set `app.py` as the main file
4. Streamlit installed the packages from `requirements.txt` and published the app

Any new commit to the repository updates the live app automatically.

---

## 🧪 Try These Reviews

| Review | Expected result |
|--------|-----------------|
| *This movie was a masterpiece. Brilliant acting and a touching story.* | Positive |
| *A complete waste of time. The plot made no sense and the acting was fake.* | Negative |
| *It was not good at all, I expected so much more.* | Negative |

### Batch analysis format

Upload a CSV with a column containing the review text, then choose that column in the app:

```csv
review
"This movie was a masterpiece."
"Terrible movie. Boring and way too long."
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML:** scikit-learn (TF-IDF, Logistic Regression, Pipeline)
- **Data:** pandas, NumPy
- **Visualization:** Plotly
- **Web app:** Streamlit
- **Training environment:** Google Colab
- **Model saving:** joblib

---

## ⚠️ Limitations

- Trained on **movie reviews**, so accuracy is lower on other topics such as food, products or tweets.
- Supports **English only**.
- It is a **binary** classifier (positive / negative) with no neutral class. Mixed reviews may show a low-confidence warning.
- It does not understand sarcasm well.
- `scikit-learn` is pinned to version **1.6.1** in `requirements.txt` because the model was saved with that version.

---

## 🔮 Future Improvements

- Add a neutral class using a 3-class dataset
- Tune hyperparameters with GridSearchCV
- Try a transformer model such as DistilBERT and compare accuracy
- Support multiple languages
- Add a REST API for the model

---

## 👩‍💻 Author

**Yamini More**
Aspiring Data Analyst | B.Tech in Computer Science Engineering

- GitHub: [github.com/yami-0-ss](https://github.com/yami-0-ss)
- LinkedIn: [linkedin.com/in/yamini-more](https://www.linkedin.com/in/yamini-more)

---

## 🙏 Acknowledgements

- Dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle), based on the Large Movie Review Dataset by Maas et al. (Stanford, 2011)
- Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/)

---

⭐ If you found this project useful, please give it a star on GitHub!
