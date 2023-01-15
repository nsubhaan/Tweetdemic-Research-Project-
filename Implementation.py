import numpy as np 
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

import pandas as pd
import seaborn as sns
 
df=pd.read_csv(r'F:\Research Project\Datasets\Nov2019_to_March2020.csv')
df.head(10)

df.Date= pd.to_datetime(df.Date).dt.date

df.to_csv('Nov2019_to_March2020.csv')


#Using Other datsets to perform the cleaning and preprocessing

b=pd.read_csv(r'F:\Research Project\Datasets\ApriltoAug2020.csv')
c=pd.read_csv(r'F:\Research Project\Datasets\Sept-Dec2020.csv')
d=pd.read_csv(r'F:\Research Project\Datasets\01-2021-04-2021.csv')
e=pd.read_csv(r'F:\Research Project\Datasets\05-2021-08-2021.csv')
f=pd.read_csv(r'F:\Research Project\Datasets\09-2021-12-2021.csv')
g=pd.read_csv(r'F:\Research Project\Datasets\01-2022-04-2022.csv')
h=pd.read_csv(r'F:\Research Project\Datasets\05-2022-08-2022.csv')
i=pd.read_csv(r'F:\Research Project\Datasets\09-2022-12-2022.csv')



def clean_text(df):
    all_tweets = list()
    lines = df["Tweet"].values.tolist()
    for text in lines:
        text = text.lower()
        
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text)
        
        emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        text = emoji.sub(r'', text)
        
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)        
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text) 
        text = re.sub(r"\'ll", " will", text)  
        text = re.sub(r"\'ve", " have", text)  
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"don't", "do not", text)
        text = re.sub(r"did't", "did not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"couldn't", "could not", text)
        text = re.sub(r"have't", "have not", text)
        
        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
        tokens = word_tokenize(text)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        words = [w for w in words if not w in stop_words]
        words = ' '.join(words)
        all_tweets.append(words)
    return all_tweets

all_tweets = clean_text(df)
all_tweets[0:20]

df.Date= pd.to_datetime(df.Date).dt.date

gb=df.groupby("Date").size()


from wordcloud import WordCloud
import matplotlib.pyplot as plt
text = df['Tweet'].values 
wordcloud = WordCloud().generate(str(text))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

dictionary=corpora.Dictionary(text)


"""text = df['User'].values 
wordcloud = WordCloud().generate(str(text))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()"""

#Frequency of words in the total texts.
sns.set(style="darkgrid")
counts = Counter(word_list).most_common(50)
counts_df = pd.DataFrame(counts)
counts_df
counts_df.columns = ['word', 'frequency']
fig, ax = plt.subplots(figsize = (12, 12))
ax = sns.barplot(y="word", x='frequency', ax = ax, data=counts_df)
plt.savefig('wordcount_bar.png')

## making frequency distribution top 10  hashtags
def find_hash(text):
    line=re.findall(r'(?<=#)\w+',text)
    return " ".join(line)
df['hash']=df['text'].apply(lambda x:find_hash(x))
temp=df['hash'].value_counts()[:][1:11]
temp= temp.to_frame().reset_index().rename(columns={'index':'Hashtag','hash':'count'})
sns.barplot(x="Hashtag",y="count", data = temp)


#sources used to tweet
pla = data['source'][data['user_location']].value_counts().sort_values(ascending=False)
explode = (0, 0.1, 0, 0,0.01) 
plt.figure(figsize=(8,8))
pla[0:5].plot(kind = 'pie',autopct='%1.1f%%',shadow=True,explode = explode)
plt.show()

#Sentiment Distribution
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set(title='Tweet Sentiments distribution', xlabel='polarity', ylabel='frequency')
sns.distplot(sentiments_time_df['polarity'], bins=30, ax=ax)
# plt.show()
plt.savefig('sentiment_distribution.png')