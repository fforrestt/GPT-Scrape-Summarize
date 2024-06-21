import openai
import pandas as pd
import time
from aws_secrets_helper import get_secrets

# Fetch secrets
secrets = get_secrets()

# Assign OpenAI API key
openai.api_key = secrets['openai_api_key']

def extract_dicts(s):
    """
    Extract dictionary-like strings from a given string.

    Parameters:
    s (str): Input string containing dictionaries.

    Returns:
    list: List of extracted dictionary strings.
    """
    stack = []
    positions = []
    dicts = []
    
    for idx, char in enumerate(s):
        if char == '{':
            stack.append(idx)
        elif char == '}':
            if stack:
                start_pos = stack.pop()
                positions.append((start_pos, idx))

    for (start, end) in positions:
        dict_str = s[start:end+1]
        dicts.append(dict_str)

    return dicts

def text_to_dict(input_text):
    """
    Convert a block of text into a dictionary.

    Parameters:
    input_text (str): Input text to be converted.

    Returns:
    dict: Converted dictionary.
    """
    lines = input_text.strip().split("\n")
    project_dict = {}
    details_dict = {}
    
    for line in lines:
        line = line.strip()
        if "Project:" in line:
            project_dict["Project"] = line.split(":")[1].strip()
        else:
            if '-' in line:
                key, value = line.split("-", 1)
                key = key.strip()
                value = value.split(":")[1].strip() if ":" in value else value.strip()
                details_dict[key] = value
    project_dict["Details"] = details_dict
    return project_dict

def perform_openai_summarizations(filtered_target_df):
    """
    Perform OpenAI summarizations on a DataFrame.

    Parameters:
    filtered_target_df (pd.DataFrame): DataFrame containing the data to be summarized.
    """
    filtered_target_df['performed'] = 0
    filtered_target_df['summary'] = ''
    filtered_target_df['projects_list'] = ''
    filtered_target_df['first summary'] = ''

    def safe_eval(expr):
        try:
            return eval(expr, {"__builtins__": None}, {})
        except Exception as e:
            print(f"Failed to eval the following string due to {e}: {expr}")

    def trim_text_to_words(text, max_words=12000):
        """
        Trim the text to a maximum of max_words.

        Parameters:
        text (str): Input text to be trimmed.
        max_words (int): Maximum number of words to retain.

        Returns:
        str: Trimmed text.
        """
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text

    def openai_api_call(model, input_text, assistant_content=None):
        """
        Make an API call to OpenAI's GPT-4 model.

        Parameters:
        model (str): The model to use for the API call.
        input_text (str): The input text for the API call.
        assistant_content (str, optional): Additional content for the assistant role.

        Returns:
        str: The response content from the API call.
        """
        messages = [
            {"role": "system", "content": "You are a construction project manager."},
            {"role": "user", "content": input_text}
        ]
        
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.1
            )
            return response['choices'][0]['message']['content']
        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                trimmed_text = trim_text_to_words(input_text)
                messages[1] = {"role": "user", "content": trimmed_text}
                response = openai.ChatCompletion.create(
                    model="gpt-4-32k",
                    messages=messages
                )
                return response['choices'][0]['message']['content']
            else:
                raise e

    def process_text(text):
        """
        Process the text to extract project summaries.

        Parameters:
        text (str): Input text to be processed.

        Returns:
        str: Extracted project summaries.
        """
        if len(str(text).split()) > 12000:
            text = ' '.join(text.split()[:12000])
        
        initial_prompt = f"""This text should contain details about construction projects in Iraq, if it does not, do not return any information. If there are multiple projects, summarize each one individually and return in a numerical list; For each unique project you are able to find, summarize that project with these things in mind:
        Location of project, What is being built, who is funding the project, amount of money being spent, category of project that is being constructed, expected completion date, and additional information (short summary). I repeat, if the information necessary is not present, just say you do not know.
        : {text}"""
        
        list_of_projects = openai_api_call("gpt-4", initial_prompt)
        print("Process completed")
        time.sleep(10)
        return list_of_projects

    def process_entries(filtered_target_df, index):
        row = filtered_target_df.loc[index]
        summaries = []
        performed = row['performed']
        first_summary = None
        
        if 'full_content' in filtered_target_df.columns and pd.notnull(row['full_content']):
            input_text = row['full_content']
        else:
            input_text = row['content']

        if performed == 0:
            first_summary = process_text(input_text)
            print("First summary completed")
            filtered_target_df.at[index, 'first summary'] = first_summary
            if first_summary and first_summary.startswith('1'):
                new_prompt = f"""I will be providing a numerical list of construction projects, if there is no list please respond with only "no info provided"; 
                                for each number in the list, fill in the following python dictionary:
                                project_details = {{'Location of project': '', 'What is being built': '', 'Who is funding the project': '', 'Amount of money being spent': '', 'Category of project that is being constructed': '', 'Expected completion date': '', 'Additional': ''}}
                                The categories for construction projects are as follows, please only use these categories when assigning the project in question a category:

                                    Residential Construction
                                    Commercial Construction
                                    Industrial Construction
                                    Infrastructure and Heavy Civil Construction
                                    Institutional and Assembly Construction
                                    Environmental Construction
                                    Agricultural Construction
                                    Recreational Construction
                                    Specialized Construction
                                    Renovation and Remodeling
                                    Utility Construction
                                    Landscape Construction
                                {first_summary}
                                """

                generated_string = openai_api_call("gpt-4", new_prompt)
                print("Second API call completed")
                dict_strings = extract_dicts(generated_string)
                
                parsed_dicts = []
                try:
                    for ds in dict_strings:
                        try:
                            parsed_dict = safe_eval(ds)
                            if parsed_dict is not None:
                                parsed_dicts.append(parsed_dict)
                        except (SyntaxError, ValueError) as e:
                            print(f"Failed to parse the following string due to {e}: {ds}")
                            
                    if not parsed_dicts:
                        raise ValueError("No valid dictionaries found")
                        
                except Exception as e:
                    print(f"Failed to extract dictionaries using the first method due to {e}. Trying the second method...")
                    parsed_dicts = [text_to_dict(generated_string)]
                    
                filtered_target_df.at[index, 'summaries'] = str(parsed_dicts)
                filtered_target_df.at[index, 'performed'] = 1
                filtered_target_df.to_excel('GPT_sums_final.xlsx', index=False)
                print(f"Completed row {index}")
                time.sleep(2)
            
        return performed, summaries, first_summary

    for index, row in filtered_target_df.iterrows():
        if all(filtered_target_df['performed'] == 1):
            print("All rows have been processed.")
            break

        performed, summaries, first_summary = process_entries(filtered_target_df, index)
    
    if 'summaries' not in filtered_target_df.columns:
        print("No projects found.")
    else:
        filtered_target_df.to_excel(r'C:\Users\forrest.fallon\Desktop\iraq_libya\dirty_data\iraq_final_summarizations.xlsx', index=False)

def openai_money_values(filtered_target_df):
    """
    Perform OpenAI calls to extract and convert money values to USD.

    Parameters:
    filtered_target_df (pd.DataFrame): DataFrame containing the data to be processed.
    """
    filtered_target_df['USD Amount'] = ''
    
    def openai_api_money_call(model, input_text, assistant_content=None):
        """
        Make an API call to OpenAI's GPT-4 model for currency conversion.

        Parameters:
        model (str): The model to use for the API call.
        input_text (str): The input text for the API call.
        assistant_content (str, optional): Additional content for the assistant role.

        Returns:
        str: The response content from the API call.
        """
        messages = [
            {"role": "system", "content": "You are a currency exchange expert."},
            {"role": "user", "content": input_text}
        ]
        
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error: {e}")
            return None

    def process_amount(amount_text):
        retry_count = 3
        retry_delay = 10

        for attempt in range(retry_count):
            prompt = f"""
            Using the following input text, please return the amount of money being spent in U.S. dollars, convert if necessary. Here are some conversion rates:
            Iraqi Dinar to USD = 0.00077
            GPD to USD = 1.25
            EUR to USD = 1.07
            JD to USD = 1.41

            Please return just the USD value in quotes, like the following:

            "$1000000"

            Above is just an example, but please make sure to return the full number, no words. If there is no amount of money present, please just return 0.
            : {amount_text}
            """

            usd_amount = openai_api_money_call("gpt-4", prompt)

            if usd_amount is not None:
                return usd_amount
            else:
                print(f"Retry {attempt + 1}/{retry_count} failed, waiting {retry_delay} seconds before retrying.")
                time.sleep(retry_delay)

        print(f"Failed to get USD amount after {retry_count} retries.")
        return None

    for index, row in filtered_target_df.iterrows():
        amount_text = row['Amount of money being spent']
        usd_amount = process_amount(amount_text)
        print(f"Row {index}: USD Amount = {usd_amount}")
        filtered_target_df.at[index, 'USD Amount'] = usd_amount
