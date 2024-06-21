import requests
import pandas as pd
import os
import logging
import ssl
import datetime
import time
from collections import Counter
from flashtext import KeywordProcessor
import openai
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_day_range(start_date, end_date):
    """
    Generate a list of day ranges between start_date and end_date.
    
    Parameters:
    start_date (datetime): Start date.
    end_date (datetime): End date.
    
    Returns:
    list: List of tuples containing day ranges.
    """
    day_ranges = []
    curr_date = start_date
    while curr_date <= end_date:
        day_ranges.append((curr_date, curr_date + datetime.timedelta(days=1)))
        curr_date += datetime.timedelta(days=1)
    return day_ranges

def month_delta(start, end):
    """
    Generate a list of dates where each date is one month apart.
    
    Parameters:
    start (datetime): Start date.
    end (datetime): End date.
    
    Returns:
    list: List of dates.
    """
    curr = start
    dates = [curr]
    while curr < end:
        curr += datetime.timedelta(days=31)
        curr = curr.replace(day=1)
        dates.append(curr)
    return dates

def get_token():
    """
    Fetch OAuth token from Synthesio API.
    
    Returns:
    dict: Token information.
    """
    url = "https://rest.synthesio.com/security/v2/oauth/token"
    body = {
        "client_id": os.getenv("SYNTHESIO_CLIENT_ID"),
        "client_secret": os.getenv("SYNTHESIO_CLIENT_SECRET"),
        "scope": "read",
        "username": os.getenv("SYNTHESIO_USERNAME"),
        "password": os.getenv("SYNTHESIO_PASSWORD"),
        "grant_type": "password",
    } 

    try:
        resp = requests.post(url, data=body)
        logging.info(resp.json())
        return resp.json()
    except ssl.SSLError as err:
        logging.error(f"SSL error: {err}")
    except Exception as err:
        logging.error(f"Get token error: {err}")

def get_dashboard_topics(token):
    """
    Fetch dashboard topics from Synthesio API.
    
    Parameters:
    token (dict): OAuth token information.
    
    Returns:
    dict: Dashboard topics.
    """
    headers = {"Authorization": "Bearer " + token["access_token"]}
    url = "https://rest.synthesio.com/api/v2/report/398775/topics"
    resp = requests.get(url, headers=headers)
    return resp.json()

def query_api(token, languages=None, period=None, sites=None):
    """
    Query Synthesio API for data.
    
    Parameters:
    token (dict): OAuth token information.
    languages (list, optional): List of languages to filter by.
    period (dict, optional): Period to filter by.
    sites (list, optional): List of sites to filter by.
    
    Returns:
    list: List of data from the API.
    """
    headers = {
        "Authorization": "Bearer " + token["access_token"],
        "Content-Type": "application/json",
    }
    filters = {'keywords': ['Iraq AND Reconstruction', 'Construction AND Iraq', 'Reconstruction efforts AND Iraq', 'Reconstruction NEAR/9 Iraq']}
    if languages:
        filters['languages'] = languages
    if period:
        filters['period'] = period 
    if sites:
        filters['media'] = {'sites': sites}      

    body = {
        "filters": filters,
        "sort": {"date": "asc"}
    }

    offset = 0
    mentions_url = f"https://rest.synthesio.com/mention/v2/reports/398775/_search?pretty&from={offset}&size=100"
    urls = []

    try:
        while True:
            logging.info(f"Fetching data from offset {offset}")
            response = requests.post(mentions_url, json=body, headers=headers)
            data = response.json()['data']
            if not data:
                break
            urls.append({"url": mentions_url, "body": body, "headers": headers})
            offset += 100
            if offset >= 50000:
                break

        data_list = []
        for url in urls:
            response = requests.post(url['url'], json=url.get('body'), headers=url['headers'])
            data_list.extend(response.json().get('data', []))

        logging.info("Synthesio data fetch completed.")
        return data_list

    except Exception as e:
        logging.error(f"Synthesio API error: {e}")
        return []

def get_month_range(start_date, end_date):
    """
    Generate a list of month ranges between start_date and end_date.
    
    Parameters:
    start_date (datetime): Start date.
    end_date (datetime): End date.
    
    Returns:
    list: List of tuples containing month ranges.
    """
    month_ranges = []
    curr_date = start_date
    while curr_date < end_date:
        if curr_date.month == 12:
            next_date = datetime.datetime(curr_date.year + 1, 1, 1)
        else:
            next_date = datetime.datetime(curr_date.year, curr_date.month + 1, 1)
        
        if next_date > end_date:
            month_ranges.append((curr_date, end_date))
        else:
            month_ranges.append((curr_date, next_date))
        
        curr_date = next_date
    
    return month_ranges

def get_synthesio_data(start_date, end_date=None):
    """
    Fetch Synthesio data for the given date range.
    
    Parameters:
    start_date (datetime): Start date.
    end_date (datetime, optional): End date. Defaults to current datetime.
    
    Returns:
    pd.DataFrame: DataFrame containing the fetched data.
    """
    if not end_date:
        end_date = datetime.datetime.now()

    token = get_token()
    data_list = []
    day_ranges = get_day_range(start_date, end_date)

    for begin_date, end_date in day_ranges:
        period = {
            "begin": begin_date.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
            "end": end_date.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        }

        for _ in range(2):
            try:
                data = query_api(token, period=period)
                data_list.extend(data)
                logging.info(f"Data count for period {begin_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}: {len(data)}")
                break
            except KeyError:
                logging.info("Sleeping for 900 seconds due to KeyError.")
                time.sleep(900)

    df = pd.DataFrame(data_list)
    df.to_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\dirty_data\Iraq_synthesio_scrape_output_test.xlsx', index=False)
    df = df[['content', 'date', 'title', 'url']]
    df.dropna(subset=['content'], inplace=True)
    df.to_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\dirty_data\Iraq_synthesio_scrape_output_test.xlsx', index=False)
    return df

def pre_filter_synthesio(df):
    """
    Pre-filter Synthesio data based on keywords and location terms.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing Synthesio data.
    
    Returns:
    pd.DataFrame: Filtered DataFrame.
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
                'Admin 2 AR': row['Admin 2 AR'],
            }

    for key, terms in terms_dict.items():
        if "Key" in terms:
            location_processor.add_keyword("Key", (key, "Key"))
            terms.remove("Key")
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

def openai_api_call(model, input_text, assistant_content=None):
    """
    Helper function to make an API call to OpenAI's GPT-4 model.
    
    Parameters:
    model (str): Model to be used.
    input_text (str): Input text for the model.
    assistant_content (str, optional): Assistant content for the model.
    
    Returns:
    str: Response from the OpenAI API.
    """
    messages = [
        {"role": "system", "content": "You are a construction project manager."},
        {"role": "user", "content": input_text}
    ]
    
    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})

    max_retries = 5

    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            return response['choices'][0]['message']['content']
        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                response = openai.ChatCompletion.create(model="gpt-4-32k", messages=messages)
                return response['choices'][0]['message']['content']
            else:
                raise e
        except (openai.error.ServiceUnavailableError, openai.error.APIError) as e:
            if "502 Bad Gateway" in str(e) or isinstance(e, openai.error.ServiceUnavailableError):
                if attempt < max_retries:
                    time.sleep(60 * attempt)
                    continue
                else:
                    logging.error(f"Max retries reached. Service is still unavailable after {attempt} attempts.")
                    raise
            else:
                raise e

def process_text(text):
    """
    Process text using OpenAI API to determine relevance to construction projects in Iraq.
    
    Parameters:
    text (str): Text to be processed.
    
    Returns:
    str: Processed response from the OpenAI API.
    """
    model_to_use = "gpt-3.5-turbo"
    initial_prompt = f"""I am about to paste some text, if that text contains information about construction, reconstruction, or large scale projects taking place, please reply with only "yes", if the text is talking about a project taking place in Iraq, please reply with only "yes, Iraq", if it does not contain any of this information please reply with only "no". : {text}"""
    try:
        response = openai_api_call(model_to_use, initial_prompt)
    except InvalidRequestError as e:
        logging.info(f"Caught an error: {e}. Switching to gpt-3.5-turbo-16k.")
        model_to_use = "gpt-3.5-turbo-16k"
        response = openai_api_call(model_to_use, initial_prompt)
    return response.strip()

def fetch_and_update(df):
    """
    Fetch and update the 'content' column with full article text from URLs in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing Synthesio data.
    
    Returns:
    pd.DataFrame: Updated DataFrame.
    """
    df['full_content'] = ''
    for index, row in df.iterrows():
        if 'yes' in row['content_check'].lower():
            url = row['url']
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                article_text = ' '.join([p.get_text() for p in paragraphs])
                df.at[index, 'full_content'] = article_text
            except Exception as e:
                logging.error(f"An error occurred while fetching {url}: {e}")

    df.loc[df['full_content'] != '', 'content'] = df['full_content']
    return df

def process_entries(row, index):
    """
    Process individual entries in the DataFrame.
    
    Parameters:
    row (pd.Series): Row of the DataFrame.
    index (int): Index of the row.
    
    Returns:
    tuple: Processed 'performed' status and 'summaries'.
    """
    performed = row['performed']
    summaries = []
    if performed == 0:
        summaries = process_text(row['content'])
        if summaries:
            performed = 1
        logging.info(f"Completed {index}")
        time.sleep(4)
    return performed, summaries

def process_dataframe(df):
    """
    Process the DataFrame by updating 'performed' status and fetching full content for relevant rows.
    
    Parameters:
    df (pd.DataFrame): DataFrame to be processed.
    
    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    if 'performed' not in df.columns:
        df['performed'] = 0

    for index, row in df.iterrows():
        if row['performed'] == 1:
            continue
        performed, summaries = process_entries(row, index)
        df.at[index, 'performed'] = performed
        df.at[index, 'content_check'] = summaries
        df.to_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\dirty_data\synthesio_filtered_sums_dirty.xlsx', index=False)
        if all(df['performed'] == 1):
            logging.info("All rows have been processed.")
            break

    df = df[df['content_check'].str.contains("yes", case=False, na=False)]
    df.to_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\dirty_data\synthesio_filtered_sums.xlsx', index=False)
    return df
