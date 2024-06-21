from collections import Counter
from flashtext import KeywordProcessor
import pandas as pd
import numpy as np

PROXIMITY_WINDOW = 15

def load_excel_data(path):
    """
    Load data from an Excel file.

    Parameters:
    path (str): Path to the Excel file.

    Returns:
    pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_excel(path, engine='openpyxl')

def construct_keyword_processor(df):
    """
    Construct a KeywordProcessor for location terms.

    Parameters:
    df (pd.DataFrame): DataFrame containing location terms.

    Returns:
    tuple: A KeywordProcessor and a dictionary of terms.
    """
    location_processor = KeywordProcessor(case_sensitive=False)

    terms_dict = {}
    for index, row in df.iterrows():
        if pd.notna(row['ID']):
            terms = [
                row['Town EN'], row['Town AR'], 
                row['Admin 3 EN'], row['Admin 3 AR'], 
                row['Admin 2 EN'], row['Admin 2 AR']
            ]
            terms = [term for term in terms if pd.notna(term)]
            terms_dict[row['ID']] = terms

    for key, terms in terms_dict.items():
        handle_special_terms(location_processor, key, terms, ["Key", "Market"])
        for term in terms:
            if isinstance(term, str):
                location_processor.add_keyword(term, (key, term))
                if '-' in term:
                    term_without_dash = term.replace('-', ' ')
                    location_processor.add_keyword(term_without_dash, (key, term_without_dash))

    return location_processor, terms_dict

def handle_special_terms(location_processor, key, terms, special_terms):
    """
    Handle special terms in the terms dictionary.

    Parameters:
    location_processor (KeywordProcessor): KeywordProcessor for location terms.
    key (str): Key for the terms dictionary.
    terms (list): List of terms.
    special_terms (list): List of special terms to handle.
    """
    for sterm in special_terms:
        if sterm in terms:
            location_processor.add_keyword(sterm, (key, sterm))
            terms.remove(sterm)

def build_location_info(df, terms_dict):
    """
    Build a dictionary containing location information.

    Parameters:
    df (pd.DataFrame): DataFrame containing location data.
    terms_dict (dict): Dictionary of terms.

    Returns:
    dict: Dictionary containing location information.
    """
    location_info = {}
    for index, row in df.iterrows():
        if pd.notna(row['ID']):
            location_info[row['ID']] = {
                'ID': row['ID'],
                'Location Type': row['Location Type'] if pd.notna(row['Location Type']) else 'Unknown',
                'Lat': row['Lat'] if pd.notna(row['Lat']) else 'Unknown',
                'Lon': row['Lon'] if pd.notna(row['Lon']) else 'Unknown',
                'Town EN': row['Town EN'] if pd.notna(row['Town EN']) else 'Unknown',
                'Town AR': row['Town AR'] if pd.notna(row['Town AR']) else 'Unknown',
                'Admin 3 EN': row['Admin 3 EN'] if pd.notna(row['Admin 3 EN']) else 'Unknown',
                'Admin 3 AR': row['Admin 3 AR'] if pd.notna(row['Admin 3 AR']) else 'Unknown',
                'Admin 2 EN': row['Admin 2 EN'] if pd.notna(row['Admin 2 EN']) else 'Unknown',
                'Admin 2 AR': row['Admin 2 AR'] if pd.notna(row['Admin 2 AR']) else 'Unknown',
                'Terms': [term for term in terms_dict.get(row['ID'], []) if pd.notna(term)]
            }
    return location_info

def extract_and_store_keywords(target_df, location_processor, location_info):
    """
    Extract and store keywords in the target DataFrame.

    Parameters:
    target_df (pd.DataFrame): DataFrame containing target data.
    location_processor (KeywordProcessor): KeywordProcessor for location terms.
    location_info (dict): Dictionary containing location information.

    Returns:
    pd.DataFrame: Updated DataFrame with matched keywords.
    """
    for index, row in target_df.iterrows():
        text_content = row['Location of project']
        found_location_keywords = location_processor.extract_keywords(text_content, span_info=True)
        handle_special_keywords(found_location_keywords, text_content, ["Key", "Market"])

        if found_location_keywords:
            keyword_ids = [keyword[0][0] for keyword in found_location_keywords]
            most_common_keyword_id, _ = Counter(keyword_ids).most_common(1)[0]
            matched_term = next((kw[0][1] for kw in found_location_keywords if kw[0][0] == most_common_keyword_id), None)
            location_data = location_info[most_common_keyword_id]

            store_matched_data(target_df, index, matched_term, location_data)

    return target_df

def handle_special_keywords(keywords, text_content, special_terms):
    """
    Handle special keywords in the keyword list.

    Parameters:
    keywords (list): List of found keywords.
    text_content (str): Text content being processed.
    special_terms (list): List of special terms to handle.
    """
    for special_term in special_terms:
        if any([kw[0][1] == special_term for kw in keywords]):
            term_position = next(((kw[1], kw[2]) for kw in keywords if kw[0][1] == special_term), None)
            if term_position:
                slice_around_term = text_content[max(0, term_position[0] - PROXIMITY_WINDOW): min(len(text_content), term_position[1] + PROXIMITY_WINDOW)]
                if "Iraq" not in slice_around_term.lower():
                    keywords[:] = [kw for kw in keywords if kw[0][1] != special_term]

def store_matched_data(df, index, matched_term, location_data):
    """
    Store matched data in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame to store the matched data.
    index (int): Index of the row to store data.
    matched_term (str): Matched term.
    location_data (dict): Dictionary containing location information.
    """
    if 'Matched ID' not in df:
        df['Matched ID'] = np.nan
    if 'Matched Term' not in df:
        df['Matched Term'] = np.nan
    if 'Matched Location Type' not in df:
        df['Matched Location Type'] = ''
    if 'Matched Lat' not in df:
        df['Matched Lat'] = ''
    if 'Matched Lon' not in df:
        df['Matched Lon'] = ''
    if 'Town EN' not in df:
        df['Town EN'] = ''
    if 'Town AR' not in df:
        df['Town AR'] = ''
    if 'Admin 3 EN' not in df:
        df['Admin 3 EN'] = ''
    if 'Admin 3 AR' not in df:
        df['Admin 3 AR'] = ''
    if 'Admin 2 EN' not in df:
        df['Admin 2 EN'] = ''
    if 'Admin 2 AR' not in df:
        df['Admin 2 AR'] = ''
    if 'ID' in location_data:
        df.at[index, 'Matched ID'] = location_data['ID']
    else:
        df.at[index, 'Matched ID'] = np.nan

    df.at[index, 'Matched Term'] = matched_term
    df.at[index, 'Matched Location Type'] = location_data['Location Type']
    df.at[index, 'Matched Lat'] = location_data['Lat']
    df.at[index, 'Matched Lon'] = location_data['Lon']
    df.at[index, 'Town EN'] = location_data['Town EN']
    df.at[index, 'Town AR'] = location_data['Town AR']
    df.at[index, 'Admin 3 EN'] = location_data['Admin 3 EN']
    df.at[index, 'Admin 3 AR'] = location_data['Admin 3 AR']
    df.at[index, 'Admin 2 EN'] = location_data['Admin 2 EN']
    df.at[index, 'Admin 2 AR'] = location_data['Admin 2 AR']

def filter_dataframe(df):
    """
    Filter the DataFrame to include only rows with matched data.

    Parameters:
    df (pd.DataFrame): DataFrame to filter.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    if 'Matched ID' not in df.columns:
        return pd.DataFrame()

    return df[
        df['Matched ID'].notna() & 
        df['Matched Term'].notna() & 
        df['Matched Location Type'].notna() & 
        df['Matched Lat'].notna() & 
        df['Matched Lon'].notna()
    ]
