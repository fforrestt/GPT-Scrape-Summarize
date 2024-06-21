# GPT-Scrape-Summarize
Python directory that scrapes various sources for text relating to chosen subject, filters content accordingly, prompts GPT to summarize relevant texts, and extracts information from GPT responses.


## Sources
### GDELT
Utilizes GDELT API calls to select articles containing one-or-many chosen keywords within a date range (currently set to -30 days to current date)
Many articles found are not of interest, however, there are enough gems hidden within to continue utilizing. Slow but very steady (rate limits necessary)

### IraqBusinessNews.com
Most reliable site that covers the topic of interest, was a god-send for this project as it has reliable sources and was scraping friendly (BeautifulSoup)

### Synthesio
Not currently part of final product, requires an enterprise account and results are few and far between. Required parsing of 100k+ rows for just a few potentially useful posts. 


## Pre-Summary Filtering
Content filtering consists of good ol' regex matching as well as using FlashText to match against a local file containing many different location/subject matter keywords of interest. 
Checks are also in place to make sure we have keywords present in the resulting rows that indicate our data is worth of spending GPT credits on. 

## GPT Summaries
Sends each row's content text through multiple GPT prompts in order to summarize, distill, and break information down to dictionary-format for python friendly extraction of information. 
Be sure to change model depending on your budget, I personally did not notice much difference between results from GPT-3.5 and GPT-4, however, GPT-4o might yield better results.
There are checks in place to keep GPT hallucinations at a minimum, but there can never be enough time spent pre-filtering your content. 

### Money Conversion
In order to consistently gather monetary value attached to each reconstruction project, attempts were made at a local regex matching solution, however, it was not trustworthy at the time of delivery.
Currently I have a prompt that reads the "Amount of money being spent" column through a GPT-4 prompt which returns a more steady stream of US currency figures (many of which are converted from other currencies in this process).


## Location Matching/General Cleanup
Post-summary and info-extraction, locations are matched against the local file (FlashText) containing thousands of location names and coordinates. This allows for the GIS team to place results to a dashboard and gather stats geographically. 
Data is then cleaned and combined/de-duplicated with previous-days dataframe. 

## S3 Download/Upload
As previous step mentioned, the previous-day's dataframe is downloaded from S3 and saved locally in order to merge with current df. Once merge is complete, final df is uploaded back to S3 with current date as suffix in file name, backup table is also created. 

## GIS Dashboard Update
Minor arcGIS script is ran in order to update online dashboard, as long as credits are available. 
