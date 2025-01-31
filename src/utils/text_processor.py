import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def process_texts_to_csv(
    input_dir: str = "data/raw_texts", output_file: str = "data/processed/texts.csv"
):
    """Process text files into a CSV with content and URLs"""

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Dictionary to map filenames to URLs
    url_mapping = {
        "university_info.txt": "https://itmo.ru/ru/university/about/history.html",
        "itmo_wiki_data.txt": "https://ru.wikipedia.org/wiki/Университет_ИТМО",
        # Add more mappings as needed
    }

    data = []
    input_path = Path(input_dir)

    for file_path in input_path.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            url = url_mapping.get(file_path.name, "https://itmo.ru")

            data.append(
                {
                    "content": content,
                    "url": url,
                    "source": url,  # Add source field to match metadata
                }
            )

            logger.info(f"Processed {file_path} with URL: {url}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")

    if not data:
        raise ValueError(f"No text files found in {input_dir}")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")

    return df


if __name__ == "__main__":
    process_texts_to_csv()
