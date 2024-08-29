import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pre-trained models
resnet_model = ResNet50(weights='imagenet')
mnist_model = load_model('models\mnist_model.h5')  # Replace with your actual MNIST model path

# Load YOLO model for object detection and localization
yolo_net = cv2.dnn.readNet("yolov3.weights", "path/to/yolov3.cfg")  # Replace with actual paths
with open("path/to/coco.names", "r") as f:  # Replace with the actual path to coco.names
    yolo_classes = [line.strip() for line in f.readlines()]
yolo_layer_names = yolo_net.getLayerNames()
yolo_output_layers = [yolo_layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]


# Function to classify general images using ResNet
def classify_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict using ResNet
    predictions = resnet_model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return decoded_predictions


# Function to detect objects and localize them using YOLO
def detect_and_localize_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(yolo_output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # You can adjust this threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    result_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        label = str(yolo_classes[class_ids[i]])
        confidence = confidences[i]
        result_boxes.append({
            'box': box,
            'label': label,
            'confidence': confidence
        })

    return result_boxes


# Function to recognize handwritten digits using MNIST model
def recognize_digit(img):
    # Assuming img is a grayscale image of a digit
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    predictions = mnist_model.predict(img)
    predicted_digit = np.argmax(predictions)

    return predicted_digit
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
from .utils import classify_image, detect_and_localize_objects, recognize_digit
from django.core.files.storage import default_storage
from django.conf import settings

@csrf_exempt
def detect_and_localize(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        absolute_file_path = f"{settings.MEDIA_ROOT}/{file_path}"

        # Read the image using OpenCV
        frame = cv2.imread(absolute_file_path)

        # Detect and localize objects
        result = detect_and_localize_objects(frame)

        return JsonResponse({'result': result})

    return JsonResponse({'error': 'Invalid request or no image uploaded'}, status=400)
