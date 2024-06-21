import re
from nltk.corpus import stopwords

def construct_near_regex(search_str, near_distance=10):
    """
    Construct a regex pattern to search for terms within a specified proximity.
    
    Parameters:
    search_str (str): Search string with terms separated by 'NEAR/'.
    near_distance (int): Maximum distance (in terms of words) between terms.
    
    Returns:
    str: Constructed regex pattern.
    """
    terms = search_str.split('NEAR/')
    regex_str = r'\b{}\b'.format(re.escape(terms[0]))

    for i in range(1, len(terms)):
        term, distance = terms[i].split(' ', 1)
        regex_str += r'(?:\W+\w+){0,' + str(distance) + r'}?\W+\b{}\b'.format(re.escape(term))

    return regex_str

def location_term_matches(entry_text, location):
    """
    Match location terms in the entry text.
    
    Parameters:
    entry_text (str): Text to search within.
    location (str): Location terms to match.
    
    Returns:
    tuple: Number of matches and list of location terms.
    """
    stop_words = set(stopwords.words('english'))
    if isinstance(location, str):
        location_terms = [term.strip() for term in location.split(',')] if ',' in location else location.split()
    else:
        location_terms = []
    location_filtered = ' '.join([term for term in location_terms if term.lower() not in stop_words])
    location_found = 0
    for term in location_filtered.split():
        if "Iraq" in term:
            term = "Iraq.*"
        if re.search(r'\b' + re.escape(term) + r'\b', entry_text, re.IGNORECASE):
            location_found += 1   
    return location_found, location_terms

def being_built_term_matches(entry_text, being_built):
    """
    Match terms related to what is being built in the entry text.
    
    Parameters:
    entry_text (str): Text to search within.
    being_built (str): Terms related to what is being built.
    
    Returns:
    tuple: Match object, number of matches, and list of being built terms.
    """
    if not isinstance(being_built, str):
        return None, 0, []

    near_regex = construct_near_regex(being_built)
    being_built_match = re.search(near_regex, entry_text, re.IGNORECASE)
    being_built_terms = being_built.split()
    being_built_found = 0
    for term in being_built_terms:
        if re.search(r'\b' + re.escape(term) + r'\b', entry_text, re.IGNORECASE):
            being_built_found += 1
    return being_built_match, being_built_found, being_built_terms

def find_matches(df):
    """
    Find and match location and being built terms in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    
    Returns:
    pd.DataFrame: DataFrame with added columns for matches.
    """
    location_matches = []
    location_matched_terms = []
    being_built_matches = []
    being_built_matched_terms = []

    for index, row in df.iterrows():
        entry_text = row.get('entry', '')
        location = row.get('Location of project', '')
        being_built = row.get('What is being built', '')

        location_found, location_terms = location_term_matches(entry_text, location)
        being_built_match, being_built_found, being_built_terms = being_built_term_matches(entry_text, being_built)

        location_matches.append(location_found / len(location_terms) >= 0.5 if location_terms else False)
        location_matched_terms.append(location_found / len(location_terms) if location_terms else 0)

        being_built_matches.append(bool(being_built_match) or (being_built_found / len(being_built_terms) >= 0.5 if being_built_terms else False))
        being_built_matched_terms.append(being_built_found / len(being_built_terms) if being_built_terms else 0)

    df['location matches'] = location_matches
    df['location matched terms'] = location_matched_terms
    df['what is being built matches'] = being_built_matches
    df['what is being built matched terms'] = being_built_matched_terms

    return df
