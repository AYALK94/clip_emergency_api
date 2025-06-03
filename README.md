# clip_emergency_api
# CLIP Medical Emergency Classifier

A FastAPI-based image classification API using OpenAI's CLIP model, hosted on Hugging Face Spaces.

## 🩺 Categories (Based on NHS Emergency Tiers)

- Category 1 – Immediate life-threatening
- Category 2 – Emergency
- Category 3 – Urgent
- Category 4 – Less urgent

## 🚀 Usage

### Deploy to Hugging Face Spaces

1. Create a new Space: New Space > choose SDK > Python.
2. Upload all files (app.py, requirements.txt, README.md).
3. Spaces auto-deploys on push.

### API Endpoint

```http
POST /classify
Content-Type: multipart/form-data
Body: image file