import pandas as pd
from tqdm import tqdm
import requests
import time
import config
import os.path

class RedditScraper():    
    
    def __init__(self):
        pass
        
            
    def retrieve_posts(self, post_limit_per_subreddit, subreddit_names):
        
        url = 'https://api.pushshift.io/reddit/search/submission'
        MAX_REDDIT_RESULTS_API = 100
        progress_bar = tqdm(total = int(post_limit_per_subreddit * len(subreddit_names)))
        fields = ['title', 'selftext', 'upvote_ratio', 'subreddit']
        json_data = []
        
        for subreddit_name in subreddit_names:
        
            submission_counter = 0
            remaining_submissions = post_limit_per_subreddit
            before = int(time.time())
            
            while (remaining_submissions > 0):
                
                params = {'subreddit' : subreddit_name, 'size' : min(remaining_submissions, MAX_REDDIT_RESULTS_API), 'before' : before}
                    
                raw_data = requests.get(url, params)
                
                time.sleep(0.05)    
                
                raw_data = raw_data.json()
                json_data.extend(raw_data['data'])
                
                submission_counter += min(remaining_submissions, MAX_REDDIT_RESULTS_API)
                progress_bar.update(min(remaining_submissions, MAX_REDDIT_RESULTS_API))
                remaining_submissions -= min(remaining_submissions, MAX_REDDIT_RESULTS_API)
                               
                before = json_data[-1]['created_utc']
        
        progress_bar.close()
                
        return pd.DataFrame(data=json_data, columns=fields)
    
    
def create_dataset(subreddit_list, post_limit, filename):

    path = os.path.join(config.DATASETS_DIR, filename)

    reddit_scraper = RedditScraper()
    reddit_scraper.retrieve_posts(post_limit_per_subreddit = post_limit, subreddit_names = subreddit_list).to_csv(path, float_format = '%.2f', index = False)
    return
    

SUBREDDIT_LIST = ['bioinformatics', 'BusinessIntelligence', 'dataengineering', 'datascience', 'learnpython', 'learnmachinelearning', 'MachineLearning', 'ProgrammerHumor', 'Python']

POST_LIMIT = 75000

FILENAME = 'data.csv'

create_dataset(subreddit_list=SUBREDDIT_LIST, post_limit=POST_LIMIT, filename=FILENAME)
