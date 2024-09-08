from transformers import BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai
from config import BLIP_MODEL_NAME, API_KEY

# Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
model.eval()

# Configure Google Generative AI
genai.configure(api_key=API_KEY)
