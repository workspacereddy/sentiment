import requests
from fastapi import FastAPI, HTTPException
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
    
    # Check if the response status is OK (200)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error from Hugging Face API")
    
    # Log the raw response to inspect its structure
    print("Raw response:", response.text)

    # Parse the response
    try:
        result = response.json()
    except ValueError:
        raise HTTPException(status_code=500, detail="Failed to parse response from Hugging Face API")

    # Check if result is not empty
    if not result:
        raise HTTPException(status_code=500, detail="No result returned from Hugging Face API")

    # Process sentiment analysis results
    sentiment_result = []
    for item in result:
        # Ensure each item has the 'label' and 'score' keys
        if 'label' in item and 'score' in item:
            sentiment_result.append({
                "sentiment": item['label'],
                "score": item['score'] * 100  # Multiply score by 100 to convert to percentage
            })
        else:
            raise HTTPException(status_code=500, detail="Invalid result format from Hugging Face API")

    return {"result": sentiment_result}
