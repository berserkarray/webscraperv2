import asyncio
import logging
import os
from pyppeteer import launch
import nest_asyncio
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# Allow nested asyncio loops in environments like Jupyter or when running Uvicorn
nest_asyncio.apply()

# Set your OpenAI API key from an environment variable (or hardcode it here)
client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)
# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# FastAPI app instance
app = FastAPI()

# Define the endpoint to which the scraped text will be posted.
POST_ENDPOINT = "https://playground.mprompto.com:3000/api/v1/demo/clients/load-cleaned-text"

# Define the request model for incoming requests
class ScrapeRequest(BaseModel):
    id: str
    url: str
    term: str

def truncate_text(text, max_chars=10000):
    """Naively truncate text to a maximum number of characters."""
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]..."
    return text

async def handle_popups(page):
    """
    Dismiss common pop-ups such as cookie consent.
    """
    cookie_selectors = [
        "button#accept-cookies",
        "button.cookie-accept",
        "button[aria-label='Accept Cookies']"
    ]
    for selector in cookie_selectors:
        try:
            await page.waitForSelector(selector, {'timeout': 5000})
            await page.click(selector)
            logging.info(f"Cookie popup dismissed using selector: {selector}")
            break
        except Exception:
            continue

    # Check for sign-in pop-ups; if one appears, abort.
    signin_selectors = [
        "button#sign-in-close",
        "button.signin-close",
        "button[aria-label='Close Sign In']"
    ]
    for selector in signin_selectors:
        try:
            await page.waitForSelector(selector, {'timeout': 5000})
            logging.error("Sign-in popup detected – aborting extraction.")
            raise Exception("Sign-in popup detected; manual intervention required.")
        except Exception:
            continue

async def extract_raw_content(page):
    """
    Extract the complete HTML and visible text from the page.
    """
    try:
        await page.waitForSelector('body', {'timeout': 10000})
        html_content = await page.content()
        text_content = await page.evaluate("() => document.body.innerText")
        return html_content, text_content
    except Exception as e:
        logging.error(f"Error during extraction: {e}")
        raise

async def analyze_with_llm(html: str, text: str, term: str) -> str:
    """
    Uses the OpenAI ChatCompletion API to analyze the page content.
    It truncates the HTML and visible text to avoid exceeding the model’s context limit.
    """
    truncated_html = truncate_text(html, max_chars=10000)
    truncated_text = truncate_text(text, max_chars=10000)

    messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert web scraper. Given the truncated HTML content and the visible text from a product page, "
                "analyze the page structure and extract all the product information as a single text block. "
                "The product is specified by the term provided by the user."
            )
        },
        {
            "role": "user",
            "content": (
                f"Product term: {term}\n\n"
                f"HTML Content:\n{truncated_html}\n\n"
                f"Visible Text:\n{truncated_text}\n\n"
                "Please return the final product information as a single, cohesive text block without extra commentary."
            )
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Change to 'gpt-4' if desired and available
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
            top_p=1
        )
        final_text = response.choices[0].message.content.strip()
        logging.info("LLM analysis complete.")
        return final_text
    except Exception as e:
        logging.error(f"Error during LLM analysis: {e}")
        raise

async def scrape_product(url: str, term: str, max_retries=3) -> str:
    """
    Scrape the specified URL by handling pop-ups, extracting content,
    and using the LLM to compile product information.
    Implements retries with exponential backoff.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            browser = await launch(
                headless=True,
                executablePath='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # Using system Chrome
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            page = await browser.newPage()
            logging.info(f"Navigating to {url}")
            await page.goto(url, {'waitUntil': 'networkidle2', 'timeout': 60000})
            await handle_popups(page)
            html_content, text_content = await extract_raw_content(page)
            await browser.close()
            
            final_output = await analyze_with_llm(html_content, text_content, term)
            if term.lower() not in final_output.lower():
                logging.warning(f"Term '{term}' not clearly found in output.")
            return final_output
        except Exception as e:
            attempt += 1
            logging.error(f"Attempt {attempt} failed: {e}")
            if attempt >= max_retries:
                raise Exception(f"Scraping failed after {max_retries} attempts. Last error: {e}")
            await asyncio.sleep(2 ** attempt)

@app.post("/scrape")
async def scrape_and_post(req: ScrapeRequest):
    """
    Endpoint that accepts an id, url, and term; performs the scraping and LLM analysis,
    and then posts the result to the external endpoint in JSON format.
    """
    try:
        scraped_text = await scrape_product(req.url, req.term)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    payload = {
        "id": req.id,
        "primary_text": scraped_text,
        "secondary_text": ""
    }
    try:
        response = requests.post(POST_ENDPOINT, json=payload)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error posting result: {e}")
    return {"message": "Scrape completed and data posted successfully", "payload": payload}