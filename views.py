import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_protect
import logging
import joblib
from django.shortcuts import render
from .forms import JobDescriptionForm
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from django.template import engines

def home(request):
    django_engine = engines['django']
    print(django_engine.engine.dirs)  
    return render(request, "index.html")


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
# tfidf_vectorizer = joblib.load(os.path.join('models', 'tfidf_vectorizer.pkl'))
# svd_model = joblib.load(os.path.join('models', 'svd_model.pkl'))
# dbscan_model = joblib.load(os.path.join('models', 'dbscan_model.pkl'))

# # Helper function for preprocessing text
# def preprocess_text(text):
#     text = re.sub("(\\d|\\W)+", " ", text.lower())
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words("english"))
#     lemmatizer = WordNetLemmatizer()
#     return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 3]
# Adjusted view for AJAX
# def job_description(request):
#     if request.method == 'POST':
#         form = JobDescriptionForm(request.POST)
#         if form.is_valid():
#             description = form.cleaned_data['description']
#             cleaned_description = preprocess_text(description)
#             vectorized_description = tfidf_vectorizer.transform([' '.join(cleaned_description)])
#             reduced_description = svd_model.transform(vectorized_description)
#             cluster_prediction = dbscan_model.fit_predict(reduced_description)
            
#             dfs = get_top_features_cluster(X_descriptions.toarray(), dbscan_model.labels_, 10)
#             cluster_top_features = dfs[cluster_prediction[0]].values.tolist()  # Assuming get_top_features_cluster returns a DataFrame
            
#             return JsonResponse({
#                 'cluster': cluster_prediction[0],
#                 'top_features': cluster_top_features
#             })
#     else:
#         form = JobDescriptionForm()
#         # Handle non-POST request somehow