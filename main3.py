import pandas as pd
from datetime import datetime, timedelta
from openai_sums import perform_openai_summarizations, openai_money_values
from gdelt_scrape import main_gdelt_process, pre_filter_gdelt
from synthesio_scrape import get_synthesio_data, pre_filter_synthesio, process_dataframe, fetch_and_update
from iraq_businessnews import fetch_iraq_business_news
from location_keyterms_check import process_data_lkc
import data_processing as dp
import text_matching as tm
import location_matching as lm
from sorting import process_data

def main():
    # Define the start and end dates for data processing
    start_date = pd.Timestamp('2010-01-01')
    end_date = pd.Timestamp(datetime.now())

    # Fetch and process GDELT data
    gdelt_df = main_gdelt_process(start_date, end_date)
    gdelt_df = pre_filter_gdelt(gdelt_df)
    gdelt_df_path = r"C:\Users\forrest.fallon\Desktop\iraq_libya\dirty_data\Iraq_gdelt_dirty.xlsx"
    gdelt_df.to_excel(gdelt_df_path, engine='openpyxl', index=False)
    gdelt_df['source'] = 'GDELT'

    # Fetch and process Iraq Business News data
    ibn_df = fetch_iraq_business_news(start_date)
    ibn_df['source'] = 'Iraq Businessnews'

    # Concatenate GDELT and Iraq Business News data
    df = pd.concat([gdelt_df, ibn_df], ignore_index=True)
    df = df[~(df == 'none').all(axis=1)]
    df.to_excel(r"C:\Users\forrest.fallon\Desktop\iraq_libya\dirty_data\merged_dirty.xlsx", index=False)

    # Perform OpenAI summarizations
    perform_openai_summarizations(df)
    print(df)

    column_name_dict = {
        'Location of project': ['Location of project', 'Location'],
        'What is being built': ['What is being built'],
        'Who is funding the project': ['Who is funding the project'],
        'Amount of money being spent': ['Amount of money being spent'],
        'Category of project that is being constructed': ['Category of project that is being constructed'],
        'Expected completion date': ['Expected completion date'],
        'Additional': ['Additional']
    }

    list_columns = [
        'Location of project',
        'What is being built',
        'Who is funding the project',
        'Amount of money being spent',
        'Category of project that is being constructed',
        'Expected completion date',
        'Additional'
    ]

    retain_columns = ['title', 'date', 'content', 'url', 'performed', 'summaries', 'source']

    try:
        # Extract and process data
        df = dp.extract_columns_from_df(df, column_name_dict)
        df.to_csv('testing.csv')
        print("df after extract_columns_from_df", df)

        final_df = dp.explode_list_columns(df, list_columns, retain_columns)
        final_df.to_csv('testing2.csv')
        print("df after explode_list_columns", final_df)

        df = dp.apply_terms_presence_check(final_df)
        df.to_csv('testing3.csv')
        print("df after apply_terms_presence", df)

        df = tm.find_matches(df)
        print("df after find_matches", df)

        ldf = lm.load_excel_data(r'C:\Users\forrest.fallon\Desktop\iraq_libya\ref_tables\Iraq_All_PlaceNames.xlsx')
        df['Location of project'] = df['Location of project'].fillna('').astype(str)

        location_processor, terms_dict = lm.construct_keyword_processor(ldf)
        location_info = lm.build_location_info(ldf, terms_dict)
        target_df2 = lm.extract_and_store_keywords(df, location_processor, location_info)
        print("target_df2 after extract_and_store_keywords", target_df2)

        filtered_df = lm.filter_dataframe(target_df2)
        print(filtered_df)

        # Process and save the final data
        process_data(filtered_df, 'master_table_current.csv')
    except Exception as e:
        print(e, "No projects found")

if __name__ == '__main__':
    main()
    
#    try:    
#     synthesio_df = get_synthesio_data(start_date, end_date)
#     synthesio_df = pre_filter_synthesio(synthesio_df)
#     synthesio_df = process_dataframe(synthesio_df)
#     synthesio_df = fetch_and_update(synthesio_df)
#     synthesio_df['source'] = 'SYNTHESIO'
#     synthesio_df = synthesio_df[['content', 'date', 'title', 'url', 'source']]
#   except Exception as e:
#     print(e, "Synthesio Failed to Scrape")
#     data = {
#         'date': ['none'],
#         'title': ['none'],
#         'content': ['none'],
#         'url': ['none']
#     }
#     synthesio_df = pd.DataFrame(data)
#     synthesio_df['source'] = 'SYNTHESIO'
