from bs4 import BeautifulSoup, Tag
import requests
import html2text
import config 
from urllib.parse import urlparse
from openai import AzureOpenAI
from requests.exceptions import RequestException

client = AzureOpenAI(azure_endpoint = config.openai_api_base, api_key= config.openai_api_key, api_version='2023-05-15')

def get_domain(url):
    result = urlparse(url)
    domain = result.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    domain_name = domain.split(".")[0]
    return domain_name


def get_description_from_iframe_url(iframe_url):
    try:
        response = requests.get(iframe_url)
    except Exception as e:
        return None
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Locate the description or other content you want to extract
    # (this will depend on the specific structure of the target page)
    description_tag = soup.find("meta", attrs={"name": "description"})
    if description_tag and description_tag.get("content"):
        return description_tag.get("content")

    # return domain name without the tld
    return get_domain(iframe_url)


def clean_markdown(content):
    system_prompt = """
    # HTMLMarkdownGPT
    
    # Role 
    You carefully provide accurate, factual, thoughtful, nuanced responses, and are brilliant at reasoning.
    
    ## Instructions
    1. Take rough html markdown content and reduce it to the most important content while adding markdown structure where possible
    2. Remove irrelevant links that don't seem to be relevant to the main content
    3. Remove all irrelevant content such as ads
    4. Remove all repetitive content
    5. Add header sections and lists and bolding whenever appropriate to make it easier to read
    6. Return the cleaned markdown content
    """
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
    ]
    r = client.chat.completions.create(
            model='gpt-4',
            messages=messages,
            max_tokens=2000
    )
    full_message = r.choices[0].message.content
    return full_message


def scrape_and_convert_to_markdown(url, smart_mode=False):
    # make url whole
    if not url.startswith("http"):
        url = "http://" + url
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except RequestException as e:
        return f"Failed to fetch URL {url}. Error: {str(e)}"

    soup = BeautifulSoup(response.text, "html.parser")
    if not soup.body:
        content_div = soup.find('div', attrs={'class': 'main clearfix'})
        if isinstance(content_div, Tag):
            return content_div.get_text(strip=True)
        else:
            return "No content found"
    if url.startswith("https://www.163.com"):
        post_body = soup.find('div', {'class': 'post_body'})
        return post_body.get_text()

    for tag in soup.find_all(["style", "script"]):
        tag.decompose()

    # Remove all image tags or links
    for img_tag in soup.find_all("img"):
        img_tag.decompose()

    iframes = soup.find_all("iframe")

    for iframe in iframes:
        # Get the src attribute from the iframe tag
        src = iframe.get("src")

        # Replace the iframe tag with a Markdown link to the URL
        if src:
            description = get_description_from_iframe_url(src)
            iframe_link = f"[Iframe Link: {src}]({src})"
            if description:
                iframe_link += f" - Description: {description}"
            iframe.replace_with(iframe_link)

    # Using html2text to convert HTML to Markdown
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    markdown = converter.handle(str(soup.body))

    if smart_mode:
        return clean_markdown(markdown)
    return markdown

def link_preview(url):
    if not url.startswith("http"):
        url = "http://" + url
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except RequestException as e:
        return f"Failed to fetch URL {url}. Error: {str(e)}"

    soup = BeautifulSoup(response.text, "html.parser")
    preview_dict = {}
    meta_tags = [('og:title', 'og:description'), ('twitter:title', 'twitter:description')]
    for title_attr, desc_attr in meta_tags:
        meta_title_tag = soup.find('meta', attrs={'name': title_attr})
        meta_description_tag = soup.find('meta', attrs={'name': desc_attr})
        if meta_title_tag and meta_description_tag:
            preview_dict['title'] = meta_title_tag['content']
            preview_dict['description'] = meta_description_tag['content']
            break

    return preview_dict
