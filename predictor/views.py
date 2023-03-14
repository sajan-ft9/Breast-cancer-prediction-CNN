from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.


def index(request):
    return render(request, 'index.html')

def predict(request):
    return render(request, 'predict.html')


def result(request):
    if request.method == "POST" and request.FILES['image']:
        from django.core.files.storage import FileSystemStorage
        import tensorflow as tf
        from django.conf import settings
        import numpy as np
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # from matplotlib import pyplot as plt
        import cv2

        upload = request.FILES.get('image')
        print(upload.name)
        ext = (upload.name).split(".")
        print(ext[1])
        if(ext[1] == "png"):
            fss = FileSystemStorage()
            file = fss.save(upload.name, upload)
            file_url = fss.url(file)
            img = cv2.imread(os.path.join(settings.MEDIA_ROOT,upload.name))
            # print(img)
            resize = tf.image.resize(img, (256, 256))

            from tensorflow.keras.models import load_model

            new_model = load_model('cancer_cnn_model.h5')
            prediction = new_model.predict(np.expand_dims(resize/255, 0))

            if prediction > 0.5:
                result = 'Predicted class is Malignant.\nMalignant the tumors are cancerous. The cells can grow and spread to other parts of the body.'
            else:
                result = 'Predicted class is Benign.\nBenign the cells are not yet cancerous, but they have the potential to become malignant consult the doctor'

            data = {
                'prediction': prediction,
                'result' : result,
                'file': upload.name,
            }
            return render(request, 'result.html', data)
        else:
            err = {'msg': "File not valid"}
            return render(request, 'predict.html', err)
    else:
        return render(request, 'predict.html')
