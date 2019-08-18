from django.urls import path, include
from .views import *

urlpatterns = [
    path('', home, name='home'),
    path('about/', About.as_view(), name='about'),
    path('chexnet/', ChexNet.as_view(), name='chexnet'),
    path('contact/', Contact.as_view(), name='contact'),
    path('chexnet/prediction', prediction, name='prediction'),

]
