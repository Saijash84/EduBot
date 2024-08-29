import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()


def classify_image(image_path):
    # Image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    # Decode the output (example with ImageNet classes)
    _, predicted = outputs.max(1)
    return predicted.item()
