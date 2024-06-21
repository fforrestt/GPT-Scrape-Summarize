import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import pandas as pd
import time

def get_absolute_url(base_url, relative_url):
    """
    Construct an absolute URL from a base URL and a relative URL.

    Parameters:
    base_url (str): The base URL.
    relative_url (str): The relative URL.

    Returns:
    str: The absolute URL.
    """
    if relative_url.startswith(("http://", "https://")):
        return relative_url
    from urllib.parse import urljoin
    return urljoin(base_url, relative_url)

def fetch_article_text(article_url, visited_urls):
    """
    Fetch the text content of an article from its URL.

    Parameters:
    article_url (str): The URL of the article.
    visited_urls (set): A set of URLs that have already been visited.

    Returns:
    str: The text content of the article, or None if the article cannot be fetched.
    """
    if article_url in visited_urls:
        return None

    print(f"Fetching content from: {article_url}")
    visited_urls.add(article_url)

    response = requests.get(article_url, headers={'User-Agent': 'Mozilla/5.0'})
    time.sleep(2)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract article text
    article_section = soup.find('section', class_='entry')
    if article_section:
        article_text = article_section.get_text()

        # Check for the "full report" link
        full_report_link = article_section.find('a', text='Click here to read the full report.')
        if full_report_link:
            return fetch_article_text(get_absolute_url(article_url, full_report_link['href']), visited_urls)
    else:
        article_text = "Not found"

    return article_text

def fetch_iraq_business_news(start_date):
    """
    Fetch business news articles from the Iraq Business News website starting from a given date.

    Parameters:
    start_date (datetime or str): The start date for fetching articles.

    Returns:
    pd.DataFrame: DataFrame containing the article dates, titles, contents, and URLs.
    """
    visited_urls = set()
    
    if not isinstance(start_date, (datetime, pd.Timestamp)):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    start_date = start_date.replace(tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)

    article_dates = []
    article_texts = []
    article_urls = []
    article_titles = []

    page_num = 1
    while True:
        start_url = f"https://www.iraq-businessnews.com/category/construction-and-engineering/page/{page_num}/"

        print(f"Fetching list of articles from: {start_url}")
        response = requests.get(start_url)
        if response.status_code != 200:
            print(f"Failed to fetch page {page_num}. Stopping.")
            break

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find articles
        articles = soup.find_all('article')
        if not articles:
            break

        # Flag to track if the scraping should move to the next page
        next_page = True
        for article in articles:
            article_link = article.find('a', rel='bookmark')
            article_date_tag = article.find('abbr', class_='date time published updated')

            if not article_link or not article_date_tag:
                continue

            article_date_str = article_date_tag.get('title')
            article_date = datetime.strptime(article_date_str, "%Y-%m-%dT%H:%M:%S%z")

            if start_date <= article_date <= end_date:
                article_text = fetch_article_text(article_link['href'], visited_urls)
                article_dates.append(article_date.replace(tzinfo=None))
                article_texts.append(article_text)
                article_urls.append(article_link['href'])
                article_titles.append(article_link.get('title'))  # Added this line for the title.
            elif article_date < start_date:
                next_page = False
                break

        if not next_page:
            break

        page_num += 1

    # Check if no articles were collected
    if not article_dates:
        article_dates = ["none"]
        article_titles = ["none"]
        article_texts = ["none"]
        article_urls = ["none"]

    df = pd.DataFrame({
        'date': article_dates,
        'title': article_titles,
        'content': article_texts,
        'url': article_urls
    })

    df.to_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\dirty_data\Iraq_Business_News.xlsx', index=False)
    return df
