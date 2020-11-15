from recommender.views import Similarity, CBRecommend, Train, CFRecommend
from django.conf.urls import url

urlpatterns = [
    url(r'^similarity', Similarity.as_view(), name='similarity'),
    url(r'^cbrecommend/(?P<id>[0-9]+)', CBRecommend.as_view(), name='cbrecommend'),
    url(r'^train', Train.as_view(), name='train'),
    url(r'^cfrecommend', CFRecommend.as_view(), name='cfrecommend'),
]


