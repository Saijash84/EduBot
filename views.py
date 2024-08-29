from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import cv2
import numpy as np
from .utils import process_uploaded_image, process_live_image_feed
from django.core.files.storage import default_storage
from django.conf import settings


@csrf_exempt
def document_upload(request):
    if request.method == 'POST' and request.FILES['document']:
        uploaded_file = request.FILES['document']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        absolute_file_path = f"{settings.MEDIA_ROOT}/{file_path}"

        # Process the uploaded image
        result = process_uploaded_image(absolute_file_path)

        return JsonResponse({'result': result})

    return JsonResponse({'error': 'Invalid request or no document uploaded'}, status=400)


@csrf_exempt
def live_image_feed(request):
    if request.method == 'GET':
        # Assume you fetch the frame from a live feed
        video_capture = cv2.VideoCapture(0)  # OpenCV method to access webcam

        success, frame = video_capture.read()

        if success:
            # Process the live image frame
            result = process_live_image_feed(frame)
            video_capture.release()  # Release the webcam
            return JsonResponse({'result': result})
        else:
            video_capture.release()  # Release the webcam
            return JsonResponse({'error': 'Failed to capture live feed'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
