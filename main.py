from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize the FastAPI app
app = FastAPI()

# Initialize the sentiment analysis pipeline
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0  # Use GPU if available
)

# Define the input data model (request body schema)
class TextInput(BaseModel):
    text: str

# Define the sentiment analysis endpoint
@app.post("/analyze_sentiment/")
async def analyze_sentiment(input_data: TextInput):
    input_text = input_data.text
    
    # Perform sentiment analysis
    result = sentiment_analysis(input_text)
    
    # Prepare the result to be returned
    sentiment_result = []
    for item in result:
        sentiment_result.append({
            "sentiment": item['label'],
            "score": item['score'] * 100
        })
    
    return {"result": sentiment_result}
