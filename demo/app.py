import gradio as gr
import os
import torch

from model_architecture import ModelArchitecture
from torchvision import transforms
from timeit import default_timer as timer
from typing import Tuple, Dict

class_names = ['glioma', 'healthy', 'meningioma', 'pituitary']

model = ModelArchitecture(3694, 1093, 562, 374)
model.load_state_dict(torch.load('brain_tumor_classifier.pth', map_location=torch.device('cpu'), weights_only=True))

transformations = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda img: transforms.functional.equalize(img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.18663395941257477, std=0.20173224806785583)
])

def predict(img) -> Tuple[Dict, float]:

    start_time = timer()

    img = transformations(img).unsqueeze(0)

    model.eval()
    with torch.inference_mode():

        pred_probs = torch.softmax(model(img), dim=1)

    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    pred_time = round(timer() - start_time, 5)

    return pred_labels_and_probs, pred_time

title = "Cutting-Edge Deep Learning Model for Precise Brain Tumor Classification"
description = """
###
Leveraging state-of-the-art convolutional neural networks (CNNs), this advanced MRI image classification model is engineered to distinguish between healthy and pathological brain states with exceptional accuracy. Designed to process high-resolution MRI scans, the model seamlessly categorizes images into four distinct classes: healthy, glioma, meningioma, and pituitary tumors, enabling precise and efficient medical image analysis.

At its core, the model harnesses deep feature extraction techniques to identify complex anatomical patterns and subtle textural variations within MRI scans. By transforming raw image data into a multidimensional representation of brain structures, it captures intricate spatial dependencies, ensuring high-fidelity classification. The network's architecture is optimized for both computational efficiency and diagnostic precision, making it a powerful asset in medical imaging.

#### Guidelines for Optimal Image Input
To ensure the highest accuracy in classification, users are advised to follow these best practices when uploading an MRI scan:
- **Background Consistency**: The MRI image should have a **black background** if any background is present.
- **Noise-Free Image**: The scan must be free of **numbers, letters, or symbols**, as these artifacts can introduce noise and affect prediction accuracy.
- **Clear and Unaltered MRI**: The image should be a direct MRI scan without additional annotations, graphical overlays, or modifications that might interfere with feature extraction.

The examples provided constitute an accurate representation of the usage of the guidelines above.

This AI-driven system represents a significant stride toward **automated radiological assessment**, offering a powerful tool for early diagnosis and aiding clinicians in making more informed decisions. With deep learning at its core, the model embodies the convergence of artificial intelligence and medical science, setting a new benchmark in **neuroimaging analysis**.
"""

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=4, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description)

demo.launch(debug=False,
            share=True)