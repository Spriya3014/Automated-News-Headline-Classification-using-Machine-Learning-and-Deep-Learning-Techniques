#  Automated News Headline Classification using Machine Learning and Deep Learning Techniques

##  Project Overview
This project aims to automatically classify **news headlines** into predefined categories such as Politics, Sports, Business, Entertainment, and Technology.  
It combines **Machine Learning** and **Deep Learning (LSTM)** techniques to achieve high accuracy in text classification.  

The system performs **complete NLP preprocessing**, applies **feature engineering**, and compares multiple models — including Random Forest, Naive Bayes, XGBoost, and LSTM — with **GridSearchCV** for hyperparameter optimization.  
The best-performing model is saved and later used to predict new or unseen headlines.

---

##  Project Workflow
1. **Data Preprocessing**
   - Tokenization  
   - Stopword Removal  
   - Lemmatization (using WordNetLemmatizer)  
   - Text normalization (lowercasing and punctuation removal)

2. **Feature Extraction**
   - Text vectorized using **TF-IDF Vectorizer** to convert text into numerical features.  

3. **Model Building**
   - Machine Learning Models:
     - Random Forest Classifier  
     - Multinomial Naive Bayes  
     - XGBoost Classifier  
   - Deep Learning Model:
     - LSTM (Long Short-Term Memory) Network

4. **Model Optimization**
   - Hyperparameter tuning using **GridSearchCV** 

5. **Model Evaluation**
   - Performance measured using **Accuracy**, **Precision**, **Recall**, **F1-score**.

6. **Model Saving**
   - The best-performing model is saved using **Pickle (.pkl)** for reuse.

7. **Prediction**
   - The saved model is loaded to predict:
     - Test dataset predictions  
     - User-input text predictions (new headlines)

---

##  Technologies Used
- **Python**
- **Pandas**, **NumPy**
- **NLTK**
- **Scikit-learn**
- **XGBoost**
- **TensorFlow / Keras**
- **Matplotlib**, **Seaborn**
- **Pickle**

---

##  Dataset
Dataset used: Custom or Kaggle News Headline Dataset  
Each record includes:
- `headline`: The text of the news headline  
- `category`: The label or class for that headline  

Data is divided into **training and test sets** to evaluate model generalization.

---

##  Model Performance
| Model | Accuracy | F1-score | Loss |
| Random Forest | 0.89 | 0.90 |
| Naive Bayes | 0.90 | 0.90 |
| XGBoost | 0.90 | 0.91 |
| LSTM | 0.87 |     | 0.38 |

---

