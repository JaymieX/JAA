import requests
import sys
from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

def extract_url_slug(url):
    """Extract the last slug from URL for filename"""
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    return path_parts[-1] if path_parts and path_parts[-1] else 'unknown'

def check_html_exists(url):
    """Check if HTML file already exists in temp directory"""
    slug = extract_url_slug(url)
    filename = f"hf_{slug}_raw.html"
    file_path = config.get_temp_file_path(filename)
    return file_path.exists(), file_path

def crawl_url(url):
    """Crawl a single URL and return the raw text content, skip if already exists"""
    # Check if HTML already exists
    exists, file_path = check_html_exists(url)
    if exists:
        print(f"HTML already exists, loading from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading existing file {file_path}: {e}")
            # Fall through to crawl again

    # Crawl the URL
    try:
        print(f"Crawling: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.text
    except requests.RequestException as e:
        print(f"Error crawling {url}: {e}")
        return None

def save_raw_content(url, content):
    """Save raw content to temp directory with proper filename"""
    if not content:
        print(f"No content to save for {url}")
        return None

    # Generate filename
    slug = extract_url_slug(url)
    filename = f"hf_{slug}_raw.html"

    # Get temp file path
    file_path = config.get_temp_file_path(filename)

    # Check if file already exists, skip saving if it does
    if file_path.exists():
        print(f"HTML file already exists: {file_path}")
        return file_path

    # Save content
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return None

def process_html_to_text(url, html_file_path):
    """Convert HTML to clean text using BeautifulSoup and save to data/raw"""
    if not html_file_path or not html_file_path.exists():
        print(f"HTML file not found for {url}")
        return None

    try:
        # Read HTML content
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract text content
        text_content = soup.get_text(separator='\n', strip=True)

        # Generate text filename
        slug = extract_url_slug(url)
        txt_filename = f"hf_{slug}_clean.txt"
        txt_file_path = config.get_raw_file_path(txt_filename)

        # Save clean text to data/raw
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)

        print(f"Processed HTML to text: {txt_file_path}")
        return txt_file_path

    except Exception as e:
        print(f"Error processing HTML to text for {url}: {e}")
        return None

def crawl_urls(urls):
    """Crawl multiple URLs and save their content, then process HTML to text"""
    results = []

    for url in urls:
        print(f"\n--- Processing: {url} ---")

        # Step 1: Get HTML content (crawl or load existing)
        content = crawl_url(url)
        if content:
            # Step 2: Save HTML to temp (or use existing)
            html_file_path = save_raw_content(url, content)
            if html_file_path:
                # Step 3: Process HTML to clean text and save to data/raw
                txt_file_path = process_html_to_text(url, html_file_path)

                if txt_file_path:
                    results.append({
                        'url': url,
                        'html_file_path': html_file_path,
                        'txt_file_path': txt_file_path,
                        'status': 'success'
                    })
                else:
                    results.append({
                        'url': url,
                        'html_file_path': html_file_path,
                        'txt_file_path': None,
                        'status': 'text_processing_failed'
                    })
            else:
                results.append({
                    'url': url,
                    'html_file_path': None,
                    'txt_file_path': None,
                    'status': 'html_save_failed'
                })
        else:
            results.append({
                'url': url,
                'html_file_path': None,
                'txt_file_path': None,
                'status': 'crawl_failed'
            })

    return results

def main():
    """Main function to orchestrate the crawling process"""
    # List of URLs to crawl (manually enter here)
    urls = [
        # Add your HuggingFace URLs here
        # Example: "https://huggingface.co/datasets/example-dataset"
        "https://huggingface.co/docs/transformers/installation",
        "https://huggingface.co/docs/transformers/quicktour",
        "https://huggingface.co/docs/transformers/models",
    ]

    if not urls:
        print("No URLs provided. Please add URLs to the 'urls' list in the main() function.")
        return

    print(f"Starting to crawl {len(urls)} URLs...")

    # Crawl all URLs
    results = crawl_urls(urls)

    # Print summary
    print(f"\n=== Crawling Summary ===")
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"Total URLs: {len(urls)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(urls) - successful}")

    # Print results
    for result in results:
        status_symbol = "O" if result['status'] == 'success' else "X"
        if result['status'] == 'success':
            print(f"{status_symbol} {result['url']}")
            print(f"   HTML: {result['html_file_path']}")
            print(f"   Text: {result['txt_file_path']}")
        else:
            print(f"{status_symbol} {result['url']} - {result['status']}")

if __name__ == "__main__":
    main()