from .models.openai_bot import OpenAIBot
import logging
import json
import requests
import asyncio


def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],  # Console handler
    )


def test_direct_usage(query):
    """Test direct usage of the OpenAI bot"""
    logger = logging.getLogger(__name__)

    logger.info("Initializing OpenAI bot...")
    bot = OpenAIBot()

    logger.info(f"Asking question: {query}")

    response = bot.generate_response(query, request_id=1)
    logger.info("Got response from bot")
    print("\nQuestion:", query)
    print(
        "\nDirect Response:",
        json.dumps(response.model_dump(), indent=2, ensure_ascii=False),
    )


def test_api_request(query):
    """Test API endpoint using requests"""
    logger = logging.getLogger(__name__)

    # API endpoint
    url = "http://localhost:8000/api/request"

    # Example request
    request_data = {"query": query, "id": 1}

    logger.info(f"Sending API request to {url}")

    try:
        response = requests.post(url, json=request_data)
        response.raise_for_status()

        print(
            "\nAPI Response:", json.dumps(response.json(), indent=2, ensure_ascii=False)
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Error making API request: {str(e)}")
        print(f"Error: {str(e)}")


def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Test both methods
    print("\n=== Testing Direct Usage ===")
    query = """ректор ITMO"""
    test_direct_usage(query)

    # print("\n=== Testing API Request ===")
    # test_api_request()


if __name__ == "__main__":
    main()
