from django.urls import path 
from . import views
urlpatterns = [
    path("", views.home, name="home"),
    path("predict-cluster/", views.get_salary_cluster, name="predict-cluster"),
    path("analyze-description", views.job_description, name='analyze-description')
]
