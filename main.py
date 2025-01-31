from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException
import logging

from src.schemas.request import PredictionRequest, PredictionResponse
from src.models.openai_bot import OpenAIBot

def configure_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/app.log')
        ]
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

configure_logging()
logger = logging.getLogger(__name__)

logger.info("=" * 50)
logger.info("🚀 Starting ITMO Assistant Application")
logger.info("=" * 50)

app = FastAPI(title="ITMO Assistant API")

logger.info("Initializing OpenAI bot and RAG engine...")
try:
    bot = OpenAIBot()
    logger.info("✨ Application initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize application: {str(e)}", exc_info=True)
    raise

@app.post("/api/request", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        logger.info(f"Processing prediction request with id: {request.id}")
        
        # Generate response using OpenAI bot
        response = bot.generate_response(
            query=request.query,
            request_id=request.id
        )
        
        logger.info(f"Successfully processed request {request.id}")
        return response
        
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Validation error for request {request.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        logger.error(f"Internal error processing request {request.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    logger.info("🌟 API is ready to handle requests")

if __name__ == "__main__":
    try:
        logger.info("Starting uvicorn server...")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_config=None  # Disable uvicorn's default logging
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise
