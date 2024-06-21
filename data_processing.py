import re
import pandas as pd

def normalize_text(text):
    """
    Normalize text by removing non-alphanumeric characters and converting to lowercase.
    
    Parameters:
    text (str): The text to normalize.
    
    Returns:
    str: Normalized text.
    """
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def check_terms_presence(row, column_name):
    """
    Check if the term from the specified column is present in the 'content' column of the row.
    
    Parameters:
    row (pd.Series): The row of the DataFrame.
    column_name (str): The column name where the term is located.
    
    Returns:
    bool: True if the term is present in the 'content' column, False otherwise.
    """
    term = row[column_name]
    entry = row['content']
    
    if pd.isna(term) or pd.isna(entry):
        return False
    
    term_norm = normalize_text(term)
    entry_norm = normalize_text(entry)
    
    return term_norm in entry_norm

def extract_column_data(string, column_names):
    """
    Extract data from a string based on a list of column names.
    
    Parameters:
    string (str): The input string to search within.
    column_names (list of str): The list of column names to search for.
    
    Returns:
    list: A list of matched data extracted from the string.
    """
    if pd.isna(string) or not isinstance(string, str):
        return []
    matches = []
    for column_name in column_names:
        pattern = f"'{re.escape(column_name)}':\s*'([^']+)'"
        local_matches = re.findall(pattern, string, re.IGNORECASE)
        matches += [match for match in local_matches if match]
    return matches if matches else []

def extract_columns_from_df(df, column_name_dict):
    """
    Extract columns from the DataFrame based on the column name dictionary.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name_dict (dict): Dictionary where keys are column names and values are lists of alternative names.
    
    Returns:
    pd.DataFrame: DataFrame with extracted columns.
    """
    for column_name, alternatives in column_name_dict.items():
        df[column_name] = df['summaries'].apply(lambda x: extract_column_data(x, alternatives))
    return df

def explode_list_columns(df, list_columns, retain_columns):
    """
    Explode list columns into separate rows in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    list_columns (list of str): List of columns that contain lists to explode.
    retain_columns (list of str): List of columns to retain in the final DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with exploded list columns.
    """
    final_df = pd.DataFrame()
    for i, row in df.iterrows():
        exploded_data = {col: pd.Series(row[col]) for col in list_columns}
        exploded_df = pd.DataFrame(exploded_data)
        for col in retain_columns:
            exploded_df[col] = row[col]
        final_df = pd.concat([final_df, exploded_df], ignore_index=True)
    return final_df

def apply_terms_presence_check(df):
    """
    Apply term presence check to the DataFrame and add results as new columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with new columns indicating term presence.
    """
    df['location_terms_present'] = df.apply(lambda row: check_terms_presence(row, 'Location of project'), axis=1)
    df['what_is_being_built_terms_present'] = df.apply(lambda row: check_terms_presence(row, 'What is being built'), axis=1)
    return df
