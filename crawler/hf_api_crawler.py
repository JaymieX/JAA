import requests
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

def extract_url_slug(url):
    """Extract the last slug from URL for filename"""
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    return path_parts[-1] if path_parts and path_parts[-1] else 'unknown'

def crawl_url(url):
    """Crawl a single URL and return the raw text content"""
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

    # Save content
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return None

def crawl_urls(urls):
    """Crawl multiple URLs and save their content"""
    results = []

    for url in urls:
        content = crawl_url(url)
        if content:
            file_path = save_raw_content(url, content)
            if file_path:
                results.append({
                    'url': url,
                    'file_path': file_path,
                    'status': 'success'
                })
            else:
                results.append({
                    'url': url,
                    'file_path': None,
                    'status': 'save_failed'
                })
        else:
            results.append({
                'url': url,
                'file_path': None,
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
        print(f"{status_symbol} {result['url']} -> {result['file_path']}")

if __name__ == "__main__":
    main()