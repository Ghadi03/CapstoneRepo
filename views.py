import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_protect
import logging
import joblib
from django.shortcuts import render
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd 
from django.template import engines
import re
from nltk.corpus import stopwords
from .utils import tokenize_and_lemmatize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

data_path = 'actualapp/job_5000.csv'
data = pd.read_csv(data_path)



def load_model():
    model_path = 'actualapp/kmeans_model1.pkl'  # Consider using an absolute path for testing
    try:
        model = joblib.load(model_path)
        if hasattr(model, 'predict'):
            logging.info("Model loaded successfully: %s", type(model))
            return model
        else:
            logging.error("Loaded object corrupted.")
            return None
    except Exception as e:
        logging.error("Failed to load model: %s", e)
        return None

def preprocess_input(min_salary, medium_salary, max_salary):
    scaler = StandardScaler()
    salaries = np.array([[min_salary, medium_salary, max_salary]])
    salaries_normalized = scaler.fit_transform(salaries)
    return salaries_normalized

def predict_cluster(min_salary, medium_salary, max_salary):
    model = load_model()
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded object is not a model")
    salaries_normalized = preprocess_input(min_salary, medium_salary, max_salary)
    cluster = model.predict(salaries_normalized)
    return cluster

@csrf_protect
def get_salary_cluster(request):
    if request.method == 'POST':
        min_salary = request.POST.get('min_salary', 0)
        medium_salary = request.POST.get('medium_salary', 0)
        max_salary = request.POST.get('max_salary', 0)
        cluster = predict_cluster(float(min_salary), float(medium_salary), float(max_salary))
        return JsonResponse({'cluster': int(cluster[0])})
    else:
        return HttpResponse('This endpoint expects a POST request.')

#---------------------------------------------------------------------------------
#Deploying description model
# Load models


def clean_text(text):
    # Remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.lower()
    return text

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 3]
    return filtered_tokens

# Apply cleaning
data['Cleaned Description'] = data['description'].apply(clean_text)

# Vectorizing with TF-IDF
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_lemmatize, max_features=1000, stop_words='english', ngram_range=(1,2))
X_descriptions = tfidf_vectorizer.fit_transform(data['Cleaned Description'])

# Perform dimensionality reduction
svd_model = TruncatedSVD(n_components=50)
X_reduced = svd_model.fit_transform(X_descriptions)

# Cluster model
dbscan_model = DBSCAN(eps=0.5, min_samples=5)
dbscan_model.fit(X_reduced)
  
def vectorize_description(cleaned_description):
    # Transform the cleaned description
    vectorized_description = tfidf_vectorizer.transform([' '.join(cleaned_description)])
    print("Description vectorized: ", vectorized_description)

    return vectorized_description

def preprocess_description(description):
    # Clean and tokenize the description
    cleaned_description = tokenize_and_lemmatize(clean_text(description))
    print("Tokenization and Lemmatization completed: ", cleaned_description)

    # Vectorize the cleaned description
    vectorized_description = vectorize_description(cleaned_description)

    # Perform dimensionality reduction
    reduced_description = svd_model.transform(vectorized_description)
    print("Dimensionality reduction completed: ", reduced_description)

    return reduced_description


@csrf_protect
def job_description(request):
    if request.method == 'POST':
        # Directly use request.POST.get('description') to fetch data sent by AJAX
        description = request.POST.get('description', '')
        if description:
            # Preprocess the description
            reduced_description = preprocess_description(description)

            # Predict the cluster
            cluster_prediction = dbscan_model.fit_predict(reduced_description)
            print("Cluster prediction completed: ", cluster_prediction)

            # Prepare your response data and convert int64 to int
            response_data = {
                'cluster': int(cluster_prediction[0]),
            }
            return JsonResponse(response_data)


def home(request):
    django_engine = engines['django']
    print(django_engine.engine.dirs)  
    return render(request, "index.html")








# def get_top_features_cluster(tfidf_array, prediction, n_feats):
#     labels = np.unique(prediction)
#     dfs = []
#     for label in labels:
#         id_temp = np.where(prediction == label)  # indices for each cluster
#         x_means = np.mean(tfidf_array[id_temp], axis=0)  # mean tf-idf value for each feature in the cluster
#         sorted_means = np.argsort(x_means)[::-1][:n_feats]  # indices with top n tf-idf values
#         features = tfidf_vectorizer.get_feature_names_out()
#         best_features = [(features[i], x_means[i]) for i in sorted_means]
#         df = pd.DataFrame(best_features, columns=['features', 'score'])
#         dfs.append(df)
#     return df
