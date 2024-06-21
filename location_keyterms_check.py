import pandas as pd
from collections import Counter
from flashtext import KeywordProcessor

def load_dataframes():
    """
    Load the keyword and place name data from Excel files.

    Returns:
    tuple: Two pandas DataFrames containing keywords and place names respectively.
    """
    keywords_df = pd.read_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\ref_tables\LLM Scraping List.xlsx')
    df = pd.read_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\ref_tables\Iraq_All_PlaceNames.xlsx', engine='openpyxl')
    return keywords_df, df

def initialize_processors():
    """
    Initialize KeywordProcessor instances for location and construction terms.

    Returns:
    tuple: Two KeywordProcessor instances.
    """
    location_processor = KeywordProcessor(case_sensitive=False)
    construction_processor = KeywordProcessor(case_sensitive=False)
    return location_processor, construction_processor

def populate_processors(location_processor, construction_processor, df, keywords_df):
    """
    Populate the KeywordProcessor instances with keywords from the data.

    Parameters:
    location_processor (KeywordProcessor): Processor for location keywords.
    construction_processor (KeywordProcessor): Processor for construction keywords.
    df (pd.DataFrame): DataFrame containing place names.
    keywords_df (pd.DataFrame): DataFrame containing keywords.

    Returns:
    tuple: Dictionary of terms and dictionary of location info.
    """
    terms_dict = {}
    location_info = {}
    for index, row in df.iterrows():
        if pd.notna(row['ID']):
            terms = [
                row['Town EN'], row['Town AR'],
                row['Admin 3 EN'], row['Admin 3 AR'],
                row['Admin 2 EN'], row['Admin 2 AR']
            ]
            terms = [term for term in terms if pd.notna(term)]
            terms_dict[row['ID']] = terms

            # Populating location_info
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
    return terms_dict, location_info

def perform_comprehensive_check(target_df, location_processor, construction_processor, location_info):
    """
    Perform a comprehensive check for keywords and update the DataFrame.

    Parameters:
    target_df (pd.DataFrame): The target DataFrame to be processed.
    location_processor (KeywordProcessor): Processor for location keywords.
    construction_processor (KeywordProcessor): Processor for construction keywords.
    location_info (dict): Dictionary of location info.
    """
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
        text_content = str(row['content'])

        # Extract keywords for locations
        found_location_keywords = location_processor.extract_keywords(text_content, span_info=True)
        # Handle the "Key" proximity check
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
        
        # Extract keywords for construction terms
        found_construction_keywords = construction_processor.extract_keywords(text_content, span_info=False)
        target_df.at[index, 'construction terms'] = ", ".join([str(x[1]) for x in found_construction_keywords])

    construction_terms_to_check = ["Project", "Construction", "Reconstruction", "Build"]

    # Create a boolean mask for rows with relevant construction terms
    mask = target_df['construction terms'].apply(lambda x: any(term in str(x).split(", ") for term in construction_terms_to_check))
    filtered_target_df = target_df[mask]
    filtered_target_df.to_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\dirty_data\Iraq_GDELT_labeled.xlsx', index=False)

def process_data_lkc(target_df_path):
    """
    Process the target DataFrame with location and construction keyword checks.

    Parameters:
    target_df_path (str): Path to the target DataFrame Excel file.
    """
    keywords_df, df = load_dataframes()
    location_processor, construction_processor = initialize_processors()
    terms_dict, location_info = populate_processors(location_processor, construction_processor, df, keywords_df)
    target_df = pd.read_excel(target_df_path, engine='openpyxl')
    perform_comprehensive_check(target_df, location_processor, construction_processor, location_info)
