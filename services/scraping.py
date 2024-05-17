import asyncio
from functools import partial

import requests
from bs4 import BeautifulSoup


def scrape_text(url: str):
    # Send a GET request to the webpage
    print(url)
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"


async def scrape_text_async(url: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(scrape_text, url=url))
