import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Base source: https://huggingface.co/docs/hub/timm#inference

# Load from Hub 🔥
model = timm.create_model(
    'hf-hub:nateraw/resnet50-oxford-iiit-pet',
    pretrained=True
)

# Set model to eval mode for inference
model.eval()

# Create Transform
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

# Get the labels from the model config
labels = model.pretrained_cfg['label_names']
top_k = min(len(labels), 5)

# Use your own image file here...
image = Image.open('boxer.jpg').convert('RGB')

# Process PIL image with transforms and add a batch dimension
x = transform(image).unsqueeze(0)

# Pass inputs to model forward function to get outputs
out = model(x)

# Apply softmax to get predicted probabilities for each class
probabilities = torch.nn.functional.softmax(out[0], dim=0)

# Grab the values and indices of top 5 predicted classes
values, indices = torch.topk(probabilities, top_k)

# Prepare a nice dict of top k predictions
predictions = [
    {"label": labels[i], "score": v.item()}
    for i, v in zip(indices, values)
]
print(predictions)

# Alternative way of transforming the image
# Source: https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m
# Get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
output = model(transforms(image).unsqueeze(0))  # unsqueeze single image into batch of 1
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
# Prepare a nice dict of top k predictions
predictions = [
    {"label": labels[i], "score": v.item()}
    for i, v in zip(indices, values)
]
print(predictions)