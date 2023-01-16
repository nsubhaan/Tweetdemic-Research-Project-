"""
Implemantation

"""
from nltk.tokenize import RegexpTokenizer #pip install nltk
from stop_words import get_stop_words #pip install stop 
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pandas as pd
import seaborn as sns

#tokenizer:
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list 
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer 
p_stemmer=PorterStemmer()

a=pd.read_csv(r'F:\Research Project\Datasets\Nov2019_to_March2020.csv')
b=pd.read_csv(r'F:\Research Project\Datasets\ApriltoAug2020.csv')
c=pd.read_csv(r'F:\Research Project\Datasets\Sept-Dec2020.csv')
d=pd.read_csv(r'F:\Research Project\Datasets\01-2021-04-2021.csv')
e=pd.read_csv(r'F:\Research Project\Datasets\05-2021-08-2021.csv')
f=pd.read_csv(r'F:\Research Project\Datasets\09-2021-12-2021.csv')
g=pd.read_csv(r'F:\Research Project\Datasets\01-2022-04-2022.csv')
h=pd.read_csv(r'F:\Research Project\Datasets\05-2022-08-2022.csv')
i=pd.read_csv(r'F:\Research Project\Datasets\09-2022-12-2022.csv')


#text = df['Tweet'].values 
text1=str(a['Tweet'].values)
text2=str(b['Tweet'].values)
text3=str(c['Tweet'].values)
text4=str(d['Tweet'].values)
text5=str(e['Tweet'].values)
text6=str(f['Tweet'].values)
text7=str(g['Tweet'].values)
text8=str(h['Tweet'].values)
text9=str(i['Tweet'].values)

doc_set=[text1,text2,text3,text4,text5,text6,text7,text8,text9]

texts=[]

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw=(i).lower()
    tokens=tokenizer.tokenize(raw)
    
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list 
    texts.append(stemmed_tokens)
    
# turn our tokenized documents into a id <-> term dictionary 
dictionary = corpora.Dictionary(texts)    

# convert tokenized documents into a document-term matrix 
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)

print(ldamodel.print_topics (num_topics=4, num_words=6))

print(ldamodel.print_topics (num_topics=4, num_words=6))

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)

#genrating LSA model

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)

dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model

print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))

l=lsa_model[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)
  print("Coherence Score",coherence_lsa)
  
  
#Topic Coherence LDA vs LSA

sns.countplot(data=coherence_lda,data_coherence_lsa, y="Topic Coherence", x="Number of topic")  

# Worclouds for each tweets data in the particular time frames

from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud().generate(text1)
# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud = WordCloud().generate(text2)
# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud = WordCloud().generate(text3)
# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud = WordCloud().generate(text4)
# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud = WordCloud().generate(text5)
# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud = WordCloud().generate(text6)
# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud = WordCloud().generate(text7)
# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud = WordCloud().generate(text8)
# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
    
wordcloud = WordCloud().generate(text9)
# Generate plot
plt.imshow(wordcloud)
plt.axis("off")
plt.show()