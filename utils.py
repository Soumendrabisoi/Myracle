import os
import cv2
import pytesseract
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import UPLOAD_FOLDER, TESSERACT_CMD, BLIP_MODEL_NAME
from werkzeug.utils import secure_filename

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
model.eval()

def process_images(files):
    image_captions = []
    extracted_texts = []

    for file in files:
        filename = secure_filename(file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            image = Image.open(file_path).convert("RGB")
            inputs = processor(image, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            image_captions.append(caption)

            image_cv = cv2.imread(file_path)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            extracted_texts.append(text.strip())
        except Exception as e:
            raise RuntimeError(f'Processing failed for {filename}: {str(e)}')

    return image_captions, extracted_texts

def generate_test_case_prompt(context, image_captions, extracted_texts):
    return (
        "Create a series of concise manual test cases based on the feature or functionality I describe, following the format provided below. Each feature should be covered by test cases that include clear and specific verification steps, addressing both positive and negative scenarios."
        f"Context: {context}\n"
        f"Image Captions: {' | '.join(image_captions)}\n"
        f"Extracted Texts: {' | '.join(extracted_texts)}"
    )
