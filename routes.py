from flask import render_template, request, jsonify
from utils import process_images, generate_test_case_prompt
from config import UPLOAD_FOLDER, TESSERACT_CMD
import google.generativeai as genai

# Configure API Key for Google Generative AI
genai.configure(GOOGLE_GENAI_API_KEY)

def index():
    return render_template('index.html')

def describe():
    files = request.files.getlist('screenshots')
    context = request.form.get('context')
    image_captions, extracted_texts = process_images(files)

    combined_text = generate_test_case_prompt(context, image_captions, extracted_texts)

    try:
        model2 = genai.GenerativeModel("gemini-1.5-flash")
        response = model2.generate_content(combined_text)
    except Exception as e:
        return jsonify({'error': f'Test case generation failed: {str(e)}'}), 500

    return response.text
