import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
async def read_root():
    return {"message": "API is working!"}

@app.post("/analyze_sentiment/")
async def analyze_sentiment(input_data: TextInput):
    input_text = input_data.text
    
    # Send the request to Hugging Face Inference API
    response = requests.post(
        "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english",
        headers={"Authorization": "Bearer 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"},
        json={"inputs": input_text}
    )
    
    # Parse the response
    result = response.json()
    sentiment_result = [{"sentiment": item['label'], "score": item['score'] * 100} for item in result]
    
    return {"result": sentiment_result}
