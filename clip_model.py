import torch
from PIL import Image
import sys
import io
sys.path.append("./CLIP-main")
import clip
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Lazy model loading
model = None
preprocess = None

def load_clip_model():
    global model, preprocess
    if model is None or preprocess is None:
        logger.info("Loading CLIP model...")
        model, preprocess = clip.load("ViT-B/32", device=device)
        logger.info("CLIP model loaded successfully.")

# Emergency categories with prompts
category_prompt_groups = {
    "Category 1 – Life-threatening emergencies": [
        "A person not breathing, unconscious, needs CPR",
        "Cardiac arrest with no pulse",
        "Severe asthma with silent chest",
        "Anaphylaxis with airway compromise",
        "Hanging with altered consciousness"
    ],
    "Category 2 – Emergency": [
        "A person having chest pain or stroke symptoms",
        "Suspected heart attack with ECG changes",
        "Mental health crisis with self-harm",
        "Sepsis with fever and low pressure"
    ],
    "Category 3 – Urgent": [
        "A person with a broken arm or dislocated shoulder",
        "Non-severe asthma wheezing",
        "Fracture without bleeding",
        "Mild head injury needing observation"
    ],
    "Category 4 – Less urgent": [
        "A person with minor illness needing transport",
        "Back pain without red flags",
        "Sprained ankle",
        "Mild allergic reaction with rash"
    ]
}

# Cache text features
text_features_cache = {}

def initialize_text_features():
    load_clip_model()
    logger.info("Initializing cached text features...")
    with torch.no_grad():
        for category, prompts in category_prompt_groups.items():
            tokens = clip.tokenize(prompts).to(device)
            text_features_cache[category] = model.encode_text(tokens)
    logger.info("Cached text features initialized.")

initialize_text_features()

def classify_image(image_bytes):
    load_clip_model()
    try:
        image = Image.open(image_bytes).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")

    image_input = preprocess(image).unsqueeze(0).to(device)
    category_scores = {}

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        for category, text_features in text_features_cache.items():
            sims = (image_features @ text_features.T).squeeze(0)
            category_scores[category] = sims.mean().item()

    best_category = max(category_scores, key=category_scores.get)
    return best_category

def match_image_to_text(image_bytes, description):
    load_clip_model()

    if not description.strip() or len(description) > 1000:
        raise ValueError("Invalid description.")

    try:
        image = Image.open(image_bytes).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")

    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([description]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        similarity = (image_features @ text_features.T).item()

    return similarity

def predict_combined(image_bytes, description=None):
    image_bytes.seek(0)
    image_for_class = image_bytes.read()
    image_bytes.seek(0)

    best_category = classify_image(io.BytesIO(image_for_class))

    if description:
        image_bytes.seek(0)
        similarity_score = match_image_to_text(image_bytes, description)

        best_score = -1
        best_matching_category = best_category

        for category, prompts in category_prompt_groups.items():
            for prompt in prompts:
                image_bytes.seek(0)
                score = match_image_to_text(image_bytes, prompt)
                if score > best_score:
                    best_score = score
                    best_matching_category = category

        return {
            "predicted_category": best_matching_category,
            "based_on": "image + description",
            "match_score": round(best_score, 3),
        }

    return {
        "predicted_category": best_category,
        "based_on": "image only"
    }            


















































































# import torch
# from PIL import Image
# import sys
# sys.path.append("./CLIP-main")  # Make sure CLIP-main folder exists here
# import clip
# import numpy as np

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# # NHS category labels
# category_prompt_groups = {
#     "Category 1": [
#         "A person not breathing or unconscious",
#         "Someone receiving CPR",
#         "Cardiac arrest with no pulse",
#         "Severe asthma with silent chest",
#         "Anaphylaxis with airway swelling",
#         "Hanging or choking with altered consciousness",
#         "Trauma with massive bleeding"
#     ],
#     "Category 2": [
#         "A person having chest pain or stroke symptoms",
#         "Suspected heart attack needing urgent care",
#         "Mental health crisis with self-harm",
#         "Sepsis with high fever and confusion",
#         "Severe abdominal pain needing hospital"
#     ],
#     "Category 3": [
#         "A person with a broken arm or bleeding",
#         "Dislocated shoulder needing emergency care",
#         "Asthma attack responding to inhaler",
#         "Head injury needing observation",
#         "Possible miscarriage without bleeding"
#     ],
#     "Category 4": [
#         "A person with minor illness needing transport",
#         "Sprained ankle or minor soft tissue injury",
#         "Mild allergic reaction with rash",
#         "Back pain flare-up with no red flags",
#         "Wound dressing follow-up"
#     ]
# }
# tokenized_prompts = clip.tokenize(category_prompt_groups).to(device)

# def classify_image(image_bytes):
#     image = Image.open(image_bytes).convert("RGB")
#     image_input = preprocess(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         image_features = model.encode_image(image_input)
#         text_features = model.encode_text(tokenized_prompts)
#         logits = image_features @ text_features.T
#         probs = logits.softmax(dim=-1).cpu().numpy()

#     best_idx = np.argmax(probs)
#     return {
#         "category": category_prompts[best_idx],
#         "confidence": float(probs[0][best_idx]),
#         "probabilities": {category_prompts[i]: float(probs[0][i]) for i in range(4)}
#     }

# def match_image_to_text(image_bytes, description):
#     image = Image.open(image_bytes).convert("RGB")
#     image_input = preprocess(image).unsqueeze(0).to(device)
#     text_input = clip.tokenize([description]).to(device)

#     with torch.no_grad():
#         image_features = model.encode_image(image_input)
#         text_features = model.encode_text(text_input)
#         similarity = (image_features @ text_features.T).item()

#     return {
#         "description": description,
#         "similarity_score": round(similarity, 3)
#     }