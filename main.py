import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for all domains (you can restrict this to specific domains as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific domains like "https://your-frontend.vercel.app"
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)



class TextInput(BaseModel):
    text: str


@app.get("/")
async def read_root():
    return {"message": "API is working!"}

@app.options("/analyze_sentiment/")
async def handle_options():
    return {"message": "OPTIONS request handled"}

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
    print("Raw response text:", response.text)

    # Parse the response
    try:
        result = response.json()
        print("Parsed response:", result)
    except ValueError:
        raise HTTPException(status_code=500, detail="Failed to parse response from Hugging Face API")

    # Check if result is not empty
    if not result:
        raise HTTPException(status_code=500, detail="No result returned from Hugging Face API")

    # Unwrap the list inside the response
    sentiment_result = []
    for item in result[0]:  # Access the first item of the list (the actual sentiment results)
        # Ensure each item has the 'label' and 'score' keys
        if 'label' in item and 'score' in item:
            sentiment_result.append({
                "sentiment": item['label'],
                "score": item['score'] * 100  # Multiply score by 100 to convert to percentage
            })
        else:
            raise HTTPException(status_code=500, detail="Invalid result format from Hugging Face API")

    return {"result": sentiment_result}
