import re
import pandas as pd
import spacy
from collections import Counter
from flashtext import KeywordProcessor
from nltk.corpus import stopwords

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
    entry = row['entry']
    
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
        df[column_name] = df['summary'].apply(lambda x: extract_column_data(x, alternatives))
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

# Load DataFrame with summaries column named 'summary'
df = pd.read_excel(r'Z:\anaconda\envs\openai\pakistan\scripts\most_recent_gdelt_sums_dirty.xlsx')
df['seendate'] = pd.to_datetime(df['seendate'])
df.rename(columns={'seendate': 'date', 'content': 'entry', 'url': 'link'}, inplace=True)
print("Before:", df.shape)
# Replace double quotes with single quotes
df['summary'] = df['summary'].str.replace('"', "'")

# List of column names to extract, with alternatives
column_name_dict = {
    'Location of project': ['Location of project', 'Location'],
    'What is being built': ['What is being built'],
    'Who is funding the project': ['Who is funding the project'],
    'Amount of money being spent': ['Amount of money being spent'],
    'Category of project that is being constructed': ['Category of project that is being constructed'],
    'Expected completion date': ['Expected completion date'],
    'Additional': ['Additional']
}

# Extract columns
df = extract_columns_from_df(df, column_name_dict)

list_columns = [
    'Location of project',
    'What is being built',
    'Who is funding the project',
    'Amount of money being spent',
    'Category of project that is being constructed',
    'Expected completion date',
    'Additional'
]

# Columns to be retained
retain_columns = ['title', 'date', 'entry', 'link', 'performed', 'summary']

# Explode list columns
final_df = explode_list_columns(df, list_columns, retain_columns)

nlp = spacy.load("en_core_web_sm")

def extract_location_entities(text):
    """
    Extract location entities from text using spaCy.

    Parameters:
    text (str): The input text.

    Returns:
    list: List of location entities.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'GPE']

# Apply NER to extract locations
final_df['ner_location'] = final_df['entry'].apply(extract_location_entities)

# Check for terms presence
final_df = apply_terms_presence_check(final_df)
final_df = final_df.fillna('NAN')

def construct_near_regex(search_str, near_distance=10):
    """
    Construct a regex pattern to find terms within a specified proximity.

    Parameters:
    search_str (str): The search string with NEAR conditions.
    near_distance (int): The proximity distance for NEAR conditions.

    Returns:
    str: The regex pattern.
    """
    terms = search_str.split(' NEAR/')
    regex_str = r'\b{}\b'.format(re.escape(terms[0]))

    for i in range(1, len(terms)):
        term, distance = terms[i].split(' ', 1)
        regex_str += r'(?:\W+\w+){0,' + str(distance) + r'}?\W+\b{}\b'.format(re.escape(term))

    return regex_str

def find_matches(df):
    """
    Find matches for location and construction terms in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with match results.
    """
    location_matches = []
    location_matched_terms = []
    being_built_matches = []
    being_built_matched_terms = []

    stop_words = set(stopwords.words('english'))

    for index, row in df.iterrows():
        entry_text = row.get('entry', '')
        location = row.get('Location of project', '')
        being_built = row.get('What is being built', '')

        location_terms = [term.strip() for term in location.split(',')] if ',' in location else location.split()
        location_filtered = ' '.join([term for term in location_terms if term.lower() not in stop_words])

        location_found = 0
        for term in location_filtered.split():
            if "Iraq" in term:
                term = "Iraq.*"
            if re.search(r'\b' + re.escape(term) + r'\b', entry_text, re.IGNORECASE):
                location_found += 1
        location_matched_terms.append(location_found / len(location_terms) if location_terms else 0)

        near_regex = construct_near_regex(being_built)
        being_built_match = re.search(near_regex, entry_text, re.IGNORECASE)

        being_built_terms = being_built.split()
        being_built_found = 0
        for term in being_built_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', entry_text, re.IGNORECASE):
                being_built_found += 1
        being_built_matched_terms.append(being_built_found / len(being_built_terms) if being_built_terms else 0)

        location_matches.append(location_found / len(location_terms) >= 0.5 if location_terms else False)
        being_built_matches.append(bool(being_built_match) or (being_built_found / len(being_built_terms) >= 0.5 if being_built_terms else False))

    df['location matches'] = location_matches
    df['location matched terms'] = location_matched_terms
    df['what is being built matches'] = being_built_matches
    df['what is being built matched terms'] = being_built_matched_terms

    return df

# Use the function
final_df = find_matches(final_df)

# FlashText for location processing
location_processor = KeywordProcessor(case_sensitive=False)

# Load the Excel file into a DataFrame
df_places = pd.read_excel(r'Z:\anaconda\envs\openai\pakistan\scripts\data\Iraq_All_PlaceNames.xlsx', engine='openpyxl')
target_df = final_df
target_df['Location of project'] = target_df['Location of project'].fillna('').astype(str)

terms_dict = {}
for index, row in df_places.iterrows():
    if pd.notna(row['ID']):
        terms = [
            row['Town EN'], row['Town AR'], 
            row['Admin 3 EN'], row['Admin 3 AR'], 
            row['Admin 2 EN'], row['Admin 2 AR']
        ]
        terms = [term for term in terms if pd.notna(term)]
        terms_dict[row['ID']] = terms

# Add terms to location_processor
for key, terms in terms_dict.items():
    for term in terms:
        if isinstance(term, str):
            location_processor.add_keyword(term, (key, term))
            if '-' in term:
                term_without_dash = term.replace('-', ' ')
                location_processor.add_keyword(term_without_dash, (key, term_without_dash))

location_info = {}
for index, row in df_places.iterrows():
    if pd.notna(row['ID']):
        location_info[row['ID']] = {
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

PROXIMITY_WINDOW = 15

for index, row in target_df.iterrows():
    text_content = row['Location of project']
    
    # Extract keywords for locations
    found_location_keywords = location_processor.extract_keywords(text_content, span_info=True)

    special_terms = ["Key", "Market"]
    for special_term in special_terms:
        if any([kw[0][1] == special_term for kw in found_location_keywords]):
            term_position = next(((kw[1], kw[2]) for kw in found_location_keywords if kw[0][1] == special_term), None)
            if term_position is not None:
                start_pos = max(0, term_position[0] - PROXIMITY_WINDOW)
                end_pos = min(len(text_content), term_position[1] + PROXIMITY_WINDOW)
                slice_around_term = text_content[start_pos:end_pos]
                if "Iraq" not in slice_around_term.lower():
                    found_location_keywords = [kw for kw in found_location_keywords if kw[0][1] != special_term]
    
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

# Filter out rows without matched information
filtered_df = target_df[
    target_df['Matched ID'].notna() & 
    target_df['Matched Term'].notna() & 
    target_df['Matched Location Type'].notna() & 
    target_df['Matched Lat'].notna() & 
    target_df['Matched Lon'].notna() & 
    target_df['Town EN'].notna() & 
    target_df['Town AR'].notna() & 
    target_df['Admin 3 EN'].notna() & 
    target_df['Admin 3 AR'].notna() & 
    target_df['Admin 2 EN'].notna() & 
    target_df['Admin 2 AR'].notna()
]

def clean_label(label_text):
    """
    Clean label text by extracting text between single quotes.

    Parameters:
    label_text (str): The label text to clean.

    Returns:
    str: Cleaned label text.
    """
    cleaned = label_text.split("'")[1] if "'" in label_text else label_text
    return cleaned

# Read the Excel file into a DataFrame
label_df = pd.read_excel('label_report (5).xlsx', sheet_name='Code Punch')

text_label_mapping = {}

# Populate the text_label_mapping dictionary
for index, row in label_df.iterrows():
    input_text = row['text']
    labels = []
    if pd.notna(row['label_0']):
        labels.append(clean_label(row['label_0']))
    if pd.notna(row['label_1']):
        labels.append(clean_label(row['label_1']))
    text_label_mapping[input_text] = labels

# Create new columns in target_df to hold matched labels
filtered_df['Matched Category 0'] = None
filtered_df['Matched Category 1'] = None

# Match input text in target_df to labels
for index, row in filtered_df.iterrows():
    input_text = row['Category of project that is being constructed']
    if input_text in text_label_mapping:
        labels = text_label_mapping[input_text]
        if len(labels) >= 1:
            filtered_df.at[index, 'Matched Category 0'] = labels[0]
        if len(labels) >= 2:
            filtered_df.at[index, 'Matched Category 1'] = labels[1]
            
print(filtered_df)

def calculate_sort_score(row):
    """
    Calculate a sort score based on the presence of important columns.

    Parameters:
    row (pd.Series): The row of the DataFrame.

    Returns:
    int: The sort score.
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
        'not specified', 'na', 'nothing', 'none', 'n/a', 'unknown',
        'not available', 'not applicable', 'unspecified',
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

# Calculate the sort score for each row and store it in a new column
filtered_df['Sort Score'] = filtered_df.apply(calculate_sort_score, axis=1)

filtered_df['date'] = pd.to_datetime(filtered_df['date'])
filtered_df['year'] = filtered_df['date'].dt.year

# Sort by 'year' in descending order first
filtered_df = filtered_df.sort_values('year', ascending=False)

# Custom sort function
def custom_sort(group):
    return group.sort_values('Sort Score', ascending=False)

# Apply custom sort within each year
filtered_df = filtered_df.groupby('year', group_keys=False).apply(custom_sort).reset_index(drop=True)

def word_to_number(word):
    """
    Convert a word representing a number to its numerical value.

    Parameters:
    word (str): The word to convert.

    Returns:
    int: The numerical value of the word.
    """
    word_number_mapping = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
    }
    return word_number_mapping.get(word.lower(), None)

def closest_currency(match, text):
    """
    Find the closest currency match to a given amount in the text.

    Parameters:
    match (re.Match): The regex match object for the amount.
    text (str): The text to search for currency matches.

    Returns:
    str: The closest currency match.
    """
    closest_currency_match = None
    closest_distance = float('inf')
    for currency_match in re.finditer(r"(USD|U\.S\. dollars|Iraqi dinars|ID|GBP|EUR|€|dinars)", text, re.IGNORECASE):
        distance = abs(match.start() - currency_match.start())
        if distance < closest_distance:
            closest_distance = distance
            closest_currency_match = currency_match.group(1).lower()
    return closest_currency_match

def extract_monetary_values(text):
    """
    Extract monetary values from text and convert them to a common currency (USD).

    Parameters:
    text (str): The input text.

    Returns:
    float or str: The extracted monetary value in USD or a message indicating no value found.
    """
    if pd.isna(text):
        return "No value found"

    dinar_to_usd = 0.00077
    gbp_to_usd = 1.25
    eur_to_usd = 1.07

    amounts = []

    for match in re.finditer(r"(\d+(\.\d{1,2})?)%\s*of\s*(\d+(\.\d{1,2})?)\s*(million|billion|thousand|USD|dollars)?", text, re.IGNORECASE):
        percentage = float(match.group(1)) / 100
        base_amount = float(match.group(3))
        unit = match.group(5)
        currency = closest_currency(match, text) if closest_currency(match, text) else 'USD'

        if unit:
            if unit.lower().startswith('m'):
                base_amount *= 1e6
            elif unit.lower().startswith('b'):
                base_amount *= 1e9
            elif unit.lower().startswith('k'):
                base_amount *= 1e3

        calculated_amount = base_amount * percentage
        amounts.append((calculated_amount, currency))

    for match in re.finditer(r"(one|two|three|four|five|six|seven|eight|nine|ten)\s*(million|billion|trillion|thousand)", text, re.IGNORECASE):
        word = match.group(1)
        unit = match.group(2)
        amount = word_to_number(word)
        currency = closest_currency(match, text) if closest_currency(match, text) else 'USD'

        if unit.lower().startswith('m'):
            amount *= 1e6
        elif unit.lower().startswith('b'):
            amount *= 1e9
        elif unit.lower().startswith('t'):
            amount *= 1e12
        elif unit.lower().startswith('k'):
            amount *= 1e3

        amounts.append((amount, currency))

    for match in re.finditer(r"(\d+(\.\d{1,2})?)\s*(B|billion|M|million|K|k|thousand|T|trillion)?", text, re.IGNORECASE):
        amount = float(match.group(1))
        unit = match.group(3)
        currency = closest_currency(match, text) if closest_currency(match, text) else 'USD'

        if unit:
            if unit.lower().startswith('m'):
                amount *= 1e6
            elif unit.lower().startswith('b'):
                amount *= 1e9
            elif unit.lower().startswith('k'):
                amount *= 1e3
            elif unit.lower().startswith('t'):
                amount *= 1e12

        amounts.append((amount, currency))

    for i, (amount, currency) in enumerate(amounts):
        if currency in ["id", "iraqi dinars", "dinars"]:
            amounts[i] = (amount * dinar_to_usd, 'USD')
        elif currency == "gbp":
            amounts[i] = (amount * gbp_to_usd, 'USD')
        elif currency == "eur":
            amounts[i] = (amount * eur_to_usd, 'USD')
        else:
            amounts[i] = (amount, 'USD')

    if amounts:
        max_amount, _ = max(amounts, key=lambda x: x[0])
        return max_amount
    else:
        return "No value found"
