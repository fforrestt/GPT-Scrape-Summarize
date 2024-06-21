import pandas as pd
import re
import numpy as np
import boto3
from datetime import datetime
from aws_secrets_helper import get_secrets
from io import BytesIO

def clean_label(label_text):
    """
    Clean the label text by extracting content between single quotes.

    Parameters:
    label_text (str): Text containing the label.

    Returns:
    str: Cleaned label text.
    """
    return label_text.split("'")[1] if "'" in label_text else label_text

def process_labels(excel_file):
    """
    Reads the Excel file and creates a dictionary mapping input texts to labels.

    Parameters:
    excel_file (str): Path to the Excel file.

    Returns:
    dict: Dictionary mapping input texts to labels.
    """
    label_df = pd.read_excel(excel_file, sheet_name='Code Punch')
    text_label_mapping = {}
    for index, row in label_df.iterrows():
        input_text = row['text']
        labels = []
        if pd.notna(row['label_0']):
            labels.append(clean_label(row['label_0']))
        if pd.notna(row['label_1']):
            labels.append(clean_label(row['label_1']))
        text_label_mapping[input_text] = labels
    return text_label_mapping

def match_labels(target_df, text_label_mapping):
    """
    Matches input text in target_df to labels and updates the target_df columns.

    Parameters:
    target_df (pd.DataFrame): DataFrame containing the target data.
    text_label_mapping (dict): Dictionary mapping input texts to labels.

    Returns:
    pd.DataFrame: Updated DataFrame with matched labels.
    """
    target_df['Matched Category 0'] = None
    target_df['Matched Category 1'] = None
    for index, row in target_df.iterrows():
        input_text = row['Category of project that is being constructed']
        if input_text in text_label_mapping:
            labels = text_label_mapping[input_text]
            if labels:
                target_df.at[index, 'Matched Category 0'] = labels[0]
                if len(labels) >= 2:
                    target_df.at[index, 'Matched Category 1'] = labels[1]
    return target_df

def simplified_regex_based_labeling(input_text, regex_mappings):
    """
    Label the input text based on regex patterns.

    Parameters:
    input_text (str): Text to be labeled.
    regex_mappings (dict): Dictionary of regex patterns and corresponding labels.

    Returns:
    str: Matched label or None if no match is found.
    """
    if not isinstance(input_text, str):
        return None
    
    for pattern, label in regex_mappings.items():
        if re.search(pattern, input_text, re.IGNORECASE):
            return label
    return None

def enhanced_match_labels(target_df, text_label_mapping, regex_mappings):
    """
    Matches input text in target_df to labels and updates the target_df columns.

    Parameters:
    target_df (pd.DataFrame): DataFrame containing the target data.
    text_label_mapping (dict): Dictionary mapping input texts to labels.
    regex_mappings (dict): Dictionary of regex patterns and corresponding labels.

    Returns:
    pd.DataFrame: Updated DataFrame with matched labels.
    """
    target_df['Matched Category 0'] = None
    target_df['Matched Category 1'] = None
    
    for index, row in target_df.iterrows():
        input_text = row['Category of project that is being constructed']
        
        if input_text in text_label_mapping:
            labels = text_label_mapping[input_text]
            if labels:
                target_df.at[index, 'Matched Category 0'] = labels[0]
                if len(labels) >= 2:
                    target_df.at[index, 'Matched Category 1'] = labels[1]
        else:
            label = simplified_regex_based_labeling(input_text, regex_mappings)
            if label:
                target_df.at[index, 'Matched Category 0'] = label
                
    return target_df

def calculate_sort_score(row):
    """
    Calculate the sort score for each row based on important columns and keywords.

    Parameters:
    row (pd.Series): Row of the DataFrame.

    Returns:
    int: Calculated sort score.
    """
    score = 0
    important_columns = [
        'Location of project',
        'What is being built',
        'Who is funding the project',
        'Amount of money being spent',
        'Category of project that is being constructed',
        'Expected completion date'
    ]
    non_important_keywords = [
        'not specified', 'na', 'nothing', 'none', 'n/a', 'unknown', 'nan',
        'not available', 'not applicable', 'unspecified', 'no info provided',
        'not mentioned', 'not provided', 'Unspecified (information not provided)', 'not given',
        'Not specified in the text', 'Not mentioned in the text', 'Information not provided',
        'Information not available in the text', 'The article does not specify', 'unclear',
    ]
    non_important_keywords = [x.lower().strip() for x in non_important_keywords]

    for col in important_columns:
        if pd.notna(row[col]):
            cell_value = str(row[col]).lower().strip()
            if cell_value not in non_important_keywords:
                score += 1

    return score

def process_dataframe(df):
    """
    Calculate sort scores, extract monetary values, and sort the dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame to be processed.

    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    df['Sort Score'] = df.apply(calculate_sort_score, axis=1)
    
    try:
        df['date'] = pd.to_datetime(df['date'])
    except ValueError:
        df['date'] = df['date'].str.replace(r'\+00:00', '', regex=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df['year'] = df['date'].dt.year
    df = df.sort_values(by=['year', 'Sort Score'], ascending=[False, False])
    
    return df

def custom_sort(group):
    """
    Custom sort function for sorting by the sort score.

    Parameters:
    group (pd.DataFrame): DataFrame group to be sorted.

    Returns:
    pd.DataFrame: Sorted DataFrame group.
    """
    return group.sort_values('Sort Score', ascending=False)

def format_date(date_str):
    """
    Format date strings to ISO 8601 format (YYYY-MM-DD).

    Parameters:
    date_str (str): Date string to be formatted.

    Returns:
    str: Formatted date string.
    """
    try:
        date_obj = datetime.strptime(date_str, '%m/%d/%y')
    except ValueError:
        try:
            date_obj = datetime.fromisoformat(date_str.replace(' ', 'T'))
        except ValueError:
            return None

    return date_obj.strftime('%Y-%m-%d')

def convert_to_float(value):
    """
    Convert monetary values to float, considering units like billion, million, trillion.

    Parameters:
    value (str): Monetary value string.

    Returns:
    float: Converted float value.
    """
    try:
        value = str(value).replace('$', '').replace(',', '').replace('"', '')
        multiplier = 1
        if 'billion' in value:
            multiplier = 1e9
            value = value.replace('billion', '')
        elif 'million' in value:
            multiplier = 1e6
            value = value.replace('million', '')
        elif 'trillion' in value:
            multiplier = 1e12
            value = value.replace('trillion', '')
        return float(value) * multiplier
    except ValueError as e:
        print(f'Error converting value "{value}": {e}')
        return None

def process_data(input_df, final_csv_object_name):
    """
    Process the input DataFrame, match labels, calculate sort scores, and upload results to S3.

    Parameters:
    input_df (pd.DataFrame): Input DataFrame to be processed.
    final_csv_object_name (str): Name of the final CSV object for S3.
    """
    secrets = get_secrets()
    s3 = boto3.client('s3', 
                      aws_access_key_id=secrets["aws_access"], 
                      aws_secret_access_key=secrets["aws_secret_id"], 
                      region_name='us-east-2')
    
    # Download the current master table
    obj = s3.get_object(Bucket='iraq-llm-storage', Key='final/master_table_current.csv')
    data = obj['Body'].read()
    df2 = pd.read_csv(BytesIO(data))
    
    regex_mappings = {
        r'residential.*': 'Residential Construction',
        r'commercial.*': 'Commercial Construction',
        r'industr.*': 'Industrial Construction',
        r'(infrastruct.*|heavy civil).*': 'Infrastructure and Heavy Civil Construction',
        r'(institutional|assembly).*': 'Institutional and Assembly Construction',
        r'environmental.*': 'Environmental Construction',
        r'(agricultural.*|crop.*)': 'Agricultural Construction',
        r'recreational.*': 'Recreational Construction',
        r'specialized.*': 'Specialized Construction',
        r'(renovation|remodeling)': 'Renovation and Remodeling',
        r'(utilit.*|energ.*|solar|gas|electric.*|water)': 'Utility Construction',
        r'landscap.*': 'Landscape Construction',
        r'multi-sectoral|multi.*': 'Multi-Sectoral'
    }
    
    text_label_mapping = process_labels(r'C:\Users\forrest.fallon\Desktop\iraq_libya\ref_tables\label_report (5).xlsx')
    updated_df = enhanced_match_labels(input_df, text_label_mapping, regex_mappings)
    processed_df = process_dataframe(updated_df)

    processed_df['USD'] = processed_df['USD Amount'].apply(convert_to_float)
    processed_df['USD'] = processed_df['USD'].astype(str)
    processed_df['USD'] = processed_df['USD'].replace('nan', '0')
    processed_df['USD'] = processed_df['USD'].fillna(0)

    print(processed_df)
    column_order = [
        'title', 'date', 'content', 'url', 'performed', 'summaries', 'Location of project', 
        'What is being built', 'Who is funding the project', 'Amount of money being spent', 
        'Category of project that is being constructed', 'Expected completion date', 'Additional', 
        'Matched Lat', 'Matched Lon', 'Town EN', 'Town AR', 'Admin 3 EN', 'Admin 3 AR', 'Admin 2 EN', 'Admin 2 AR', 
        'Matched ID', 'Matched Term', 'Matched Location Type', 'Sort Score', 'year', 
        'Matched Category 0', 'Matched Category 1', 'USD'
    ]

    processed_df = processed_df[column_order]
    processed_df['date'] = processed_df['date'].astype(str).apply(format_date)
    processed_df = processed_df.astype(str)
    processed_df.drop_duplicates(keep='first', inplace=True)
    
    combined_df = pd.concat([processed_df, df2], ignore_index=True)
    combined_df = combined_df[column_order]
    combined_df.drop_duplicates(inplace=True)
    
    columns_to_check_for_deduplication = [
        'title', 'date', 'content', 'url', 'performed', 'summaries', 
        'Location of project', 'What is being built', 'Who is funding the project',
        'Amount of money being spent', 'Category of project that is being constructed',
        'Expected completion date', 'Additional'
    ]
    combined_df.drop_duplicates(subset=columns_to_check_for_deduplication, keep='first', inplace=True)
    combined_df['date'] = combined_df['date'].astype(str).apply(format_date)
    combined_df = combined_df.sort_values(by=['year', 'Sort Score'], ascending=[False, False])
    
    local_final_csv_path = 'C:\\Users\\forrest.fallon\\Desktop\\iraq_libya\\final_data\\master_table_current.csv'
    combined_df.to_csv(local_final_csv_path, index=False)

    today_str = datetime.today().strftime('%m%d%y')
    s3.upload_file(Filename=local_final_csv_path, 
                   Bucket='iraq-llm-storage', 
                   Key=f"final/master_table_{today_str}.csv")

    s3.upload_file(Filename=local_final_csv_path, 
                   Bucket='iraq-llm-storage', 
                   Key='final/master_table_current.csv')
