import pandas as pd
import requests
from datetime import datetime
from newspaper import Article
import json
from flashtext import KeywordProcessor
from collections import Counter
import time

def get_url_data(url: str) -> str:
    """
    Fetch data from the given URL.

    Parameters:
    url (str): The URL to fetch data from.

    Returns:
    str: The response text if the request is successful, otherwise an empty string.
    """
    try:
        resp = requests.get(url=url, timeout=10)
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"Error in get_url_data() status code: {resp.status_code}")
            return ""
    except Exception as err:
        print(f"Error in get_url_data(): {err}")
        return ""

def get_full_articles(news: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve the full content of articles from their URLs.

    Parameters:
    news (pd.DataFrame): DataFrame containing article URLs.

    Returns:
    pd.DataFrame: DataFrame with an additional 'content' column containing the article text.
    """
    urls = news["url"].values.tolist()
    res = []
    for url in urls:
        article = Article(url)
        try:
            article.download()
        except Exception as e:
            print(f'Failed to download article at {url}: {str(e)}')
            res.append(None)
            continue
        
        try:
            article.parse()
        except Exception as e:
            print(f'Failed to parse article at {url}: {str(e)}')
            res.append(None)
            continue

        res.append(article.text)

    news.insert(1, "content", res)
    return news

def scrape_gdelt(query_string: str, start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
    """
    Scrape GDELT data based on a query string and date range.

    Parameters:
    query_string (str): The query string to search for.
    start_datetime (datetime): The start datetime for the search.
    end_datetime (datetime): The end datetime for the search.

    Returns:
    pd.DataFrame: DataFrame containing the scraped GDELT data.
    """
    parameters = {
        'query': query_string,
        'mode': 'ArtList',
        'maxrecords': 250,
        'sort': 'DateDesc',
        'format': 'json',
        'startdatetime': start_datetime.strftime("%Y%m%d%H%M%S"),
        'enddatetime': end_datetime.strftime("%Y%m%d%H%M%S")
    }
    
    endpoint_url = 'https://api.gdeltproject.org/api/v2/doc/doc'
    
    # For debugging: print the URL and parameters
    print(f"URL: {endpoint_url}")
    print(f"Parameters: {parameters}")
    
    res = requests.get(url=endpoint_url, params=parameters)
    
    # For debugging: print the status code and response text
    print(f"Status Code: {res.status_code}")
    print(f"Response Text: {res.text[:500]}")  # truncate to avoid too much output
    time.sleep(5)
    
    if res.status_code == 200 and res.text.strip():
        try:
            json_res = res.json()
            if 'articles' in json_res:
                articles = json_res['articles']
                df = pd.DataFrame.from_records(articles)
                return df
            else:
                print(f"Unexpected JSON structure: {json_res}")
                return pd.DataFrame()
        except json.JSONDecodeError:
            print(f'Error decoding JSON for query "{query_string}" from {start_datetime} to {end_datetime}.')
            return pd.DataFrame()
    else:
        print(f'No data returned for query "{query_string}" from {start_datetime} to {end_datetime}.')
        return pd.DataFrame()

def parse_gdelt_list(gdelt_queries, start_time, end_time):
    """
    Parse a list of GDELT queries and return the combined results.

    Parameters:
    gdelt_queries (list of str): List of query strings to search for.
    start_time (datetime): The start datetime for the search.
    end_time (datetime): The end datetime for the search.

    Returns:
    pd.DataFrame: Combined DataFrame containing the results of the GDELT queries.
    """
    all_results_list = []
    for query in gdelt_queries:
        this_result = scrape_gdelt(query_string=query, start_datetime=start_time, end_datetime=end_time)
        if not this_result.empty:
            this_result = this_result.drop_duplicates(subset=["title"])
            this_result = get_full_articles(this_result)
            all_results_list.append(this_result)
        else:
            print("No results found for this query.")
    result_df = pd.concat(all_results_list, ignore_index=True)
    return result_df

def main_gdelt_process(start_date, end_date):
    """
    Main process to fetch and process GDELT data over a date range.

    Parameters:
    start_date (datetime): The start date for fetching data.
    end_date (datetime): The end date for fetching data.

    Returns:
    pd.DataFrame: DataFrame containing the processed GDELT data.
    """
    gdelt_queries = [
        "Iraq Reconstruction",
        "Reconstruction efforts in Iraq",
        "Reconstruction happening in Iraq",
        "Iraq Rebuilding",
        "Reconstruction plans in Iraq",
        "Plans for rebuilding in Iraq",
        "Renovations in Iraq",
        "Iraq reconstruction taking place",
        "مشروع",
        "مشاريع",
        "المشروع",
        "المشاريع",
        "بمشروع",
        "بمشاريع",
        "بالمشروع",
        "بالمشاريع",
        "لمشروع",
        "لمشاريع",
        "للمشروع",
        "للمشاريع",
        "اعمار",
        "الاعمار",
        "باعمار",
        "بالاعمار",
        "لاعمار",
        "للاعمار",
        "بناء",
        "البناء",
        "ببناء",
        "بالبناء",
        "لبناء",
        "للبناء"
    ]
    
    Iraq_keywords_df = pd.read_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\ref_tables\LLM Scraping List.xlsx', sheet_name='Sheet2')
    english_terms = Iraq_keywords_df["English"].dropna().tolist()
    english_terms = list(set(english_terms))
    arabic_terms = Iraq_keywords_df["ARABIC String"].dropna().tolist()
    arabic_terms = list(set(arabic_terms))
    
    # Create the English term combinations
    for term in english_terms:
        gdelt_queries.extend([
            f"Iraq NEAR {term}",
            f"{term} NEAR Iraq",
            f"Iraq AND {term}",
            f"{term} AND Iraq"
        ])
    
    # Create the Arabic term combinations
    for term in arabic_terms:
        gdelt_queries.extend([
            f"Iraq NEAR {term}",
            f"{term} NEAR Iraq",
            f"Iraq AND {term}",
            f"{term} AND Iraq"
        ])
    
    target_df = pd.DataFrame()
    
    while start_date <= end_date:
        end_interval = start_date + pd.Timedelta(days=2)
        print(f"Fetching data from {start_date} to {end_interval}")
        result_df = parse_gdelt_list(gdelt_queries, start_date, end_interval)
        if not result_df.empty:
            target_df = pd.concat([target_df, result_df], ignore_index=True)
        
        start_date += pd.Timedelta(days=2)
    
    target_df = target_df[['url', 'content', 'title', 'seendate']]
    target_df['seendate'] = pd.to_datetime(target_df['seendate'], format='%Y%m%dT%H%M%SZ').dt.date
    target_df.rename(columns={'seendate': 'date'}, inplace=True)
    target_df.to_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\dirty_data\Iraq_GDELT_temp.xlsx', index=False)

    return target_df

def pre_filter_gdelt(df):
    """
    Pre-filter GDELT data based on location and construction terms.

    Parameters:
    df (pd.DataFrame): DataFrame containing GDELT data.

    Returns:
    pd.DataFrame: Filtered DataFrame based on location and construction terms.
    """
    keywords_df = pd.read_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\ref_tables\LLM Scraping List.xlsx')
    locs_df = pd.read_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\ref_tables\Iraq_All_PlaceNames.xlsx', engine='openpyxl')
    target_df = df.dropna(subset=['content'])

    location_processor = KeywordProcessor(case_sensitive=False)
    construction_processor = KeywordProcessor(case_sensitive=False)

    terms_dict = {}
    location_info = {}

    for index, row in locs_df.iterrows():
        if pd.notna(row['ID']):
            terms = [
                row['Town EN'], row['Town AR'],
                row['Admin 3 EN'], row['Admin 3 AR'],
                row['Admin 2 EN'], row['Admin 2 AR']
            ]
            terms = [term for term in terms if pd.notna(term)]
            terms_dict[row['ID']] = terms

            location_info[row['ID']] = {
                'Location Type': row.get('Location Type', None),
                'Lat': row.get('Lat', None),
                'Lon': row.get('Lon', None),
                'Town EN': row['Town EN'],
                'Town AR': row['Town AR'],
                'Admin 3 EN': row['Admin 3 EN'],
                'Admin 3 AR': row['Admin 3 AR'],
                'Admin 2 EN': row['Admin 2 EN'],
                'Admin 2 AR': row['Admin 2 AR']
            }
    
    for key, terms in terms_dict.items():
        for term in terms:
            if isinstance(term, str):
                location_processor.add_keyword(term, (key, term))
                if '-' in term:
                    term_without_dash = term.replace('-', ' ')
                    location_processor.add_keyword(term_without_dash, (key, term_without_dash))

    for index, row in keywords_df.iterrows():
        arabic_keywords = row['ARABIC String'].split(' OR ')
        english_equivalent = row['English']
        keyword_type = row['Type']
        for keyword in arabic_keywords:
            construction_processor.add_keyword(keyword, (keyword_type, english_equivalent))

    target_df['Matched ID'] = None
    target_df['Matched Term'] = None
    target_df['Matched Location Type'] = None
    target_df['Matched Lat'] = None
    target_df['Matched Lon'] = None
    target_df['Town EN'] = None
    target_df['Town AR'] = None
    target_df['Admin 3 EN'] = None
    target_df['Admin 3 AR'] = None
    target_df['Admin 2 EN'] = None
    target_df['Admin 2 AR'] = None
    target_df['construction terms'] = None

    PROXIMITY_WINDOW = 15

    for index, row in target_df.iterrows():
        text_content = row['content']
        
        found_location_keywords = location_processor.extract_keywords(text_content, span_info=True)
        
        if any([kw[0][1] == "Key" for kw in found_location_keywords]):
            key_position = next(((kw[1], kw[2]) for kw in found_location_keywords if kw[0][1] == "Key"), None)
            if key_position is not None:
                start_pos = max(0, key_position[0] - PROXIMITY_WINDOW)
                end_pos = min(len(text_content), key_position[1] + PROXIMITY_WINDOW)
                slice_around_key = text_content[start_pos:end_pos]
                if "Iraq" not in slice_around_key.lower():
                    found_location_keywords = [kw for kw in found_location_keywords if kw[0][1] != "Key"]
        
        if found_location_keywords:
            keyword_ids = [keyword[0][0] for keyword in found_location_keywords]
            most_common_keyword_id, _ = Counter(keyword_ids).most_common(1)[0]
            matched_term = next((kw[0][1] for kw in found_location_keywords if kw[0][0] == most_common_keyword_id), None)
            location_data = location_info[most_common_keyword_id]
            target_df.at[index, 'Matched ID'] = most_common_keyword_id
            target_df.at[index, 'Matched Term'] = matched_term
            target_df.at[index, 'Matched Location Type'] = location_data['Location Type']
            target_df.at[index, 'Matched Lat'] = location_data['Lat']
            target_df.at[index, 'Matched Lon'] = location_data['Lon']
            target_df.at[index, 'Town EN'] = location_data['Town EN']
            target_df.at[index, 'Town AR'] = location_data['Town AR']
            target_df.at[index, 'Admin 3 EN'] = location_data['Admin 3 EN']
            target_df.at[index, 'Admin 3 AR'] = location_data['Admin 3 AR']
            target_df.at[index, 'Admin 2 EN'] = location_data['Admin 2 EN']
            target_df.at[index, 'Admin 2 AR'] = location_data['Admin 2 AR']
        
        found_construction_keywords = construction_processor.extract_keywords(text_content, span_info=False)
        target_df.at[index, 'construction terms'] = ", ".join([str(x[1]) for x in found_construction_keywords])

    construction_terms_to_check = ["Project", "Construction", "Reconstruction", "Build"]
    mask = target_df['construction terms'].apply(lambda x: any(term in str(x).split(", ") for term in construction_terms_to_check))
    filtered_target_df = target_df[mask]

    return filtered_target_df
