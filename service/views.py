from django.shortcuts import render
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
#from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse

import os
from .chexnet.model import make_predict
current_dir = os.path.dirname(os.path.abspath(__file__))

from PIL import Image
import numpy as np


from django.views.generic import TemplateView, ListView, CreateView


def home(request):
    return render(request, 'service/home.html')

class About(TemplateView):
    template_name = 'service/about.html'

class ChexNet(LoginRequiredMixin, TemplateView):
    template_name = 'service/chexnet.html'

class Contact(TemplateView):
    template_name = 'service/contact.html'
    
def prediction(request):
    if request.method == 'POST':
       # if not os.path.isdir(os.path.join(current_dir, 'uploads')):
         #   os.mkdir(os.path.join(current_dir, '..', 'uploads'))

       # path = os.path.join(current_dir, 'uploads', str(request.FILES['image']))

       # with open(path, 'wb+') as destination:
        #    for chunk in request.FILES['image'].chunks():
       ##         destination.write(chunk)
        path = request.FILES['image']
        img = Image.open(path).convert("RGB")

        img = img.resize((312, 312), Image.BILINEAR)
        heatmap, probabilities, diagnosis = make_predict(img)
        heatmap = heatmap.decode('utf8')
        print(diagnosis)
        return render(request, 'service/prediction.html', context={'heatmap': heatmap,
                                                                   'proba': probabilities,
                                                                   'diagnosis': diagnosis[0],
                                                                   'probability': diagnosis[1]})
    
    return HttpResponse("Failed")
