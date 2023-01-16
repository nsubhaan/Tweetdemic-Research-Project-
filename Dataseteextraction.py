
#Datset Extraction

import snscrape.modules.twitter as sntwitter
import pandas as pd

#Examples of extraction based on modifications of the query as per the dates

#query = "UK" and "(covid19 OR omicron OR coronavirus OR covid OR delta OR nhsuk OR govuk OR pandemic OR endemic) min_faves:100 lang:en until:2021-04-30 since:2021-01-01"
#query = "UK" and "(covid19 OR omicron OR coronavirus OR covid OR delta OR nhsuk OR govuk OR pandemic OR endemic) min_faves:500 lang:en until:2021-04-31 since:2021-01-01"
#query="(uk) min_faves:100 lang:en until:2021-04-31 since:2021-01-01"
query="(#uk) min_faves:200 lang:en until:2022-12-01 since:2022-09-01"

tweets = []
limit = 1000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.id,tweet.date, tweet.username, tweet.content,tweet.source,tweet.likeCount,tweet.replyCount,tweet.retweetCount])
        
df = pd.DataFrame(tweets, columns=['twitterid','Date', 'User', 'Tweet','source','likes','replycount','retweets'])
print(df)

 # to save to csv
df.to_csv('09-2022-12-2022.csv')