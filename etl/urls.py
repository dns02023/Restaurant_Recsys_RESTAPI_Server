from django.conf.urls import url
from .views import ReviewETL, TrainETL

urlpatterns = [
    url(r'^reviewetl', ReviewETL.as_view(), name='reviewetl'),
    url(r'^trainetl', TrainETL.as_view(), name='trainetl'),

]