#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import unicode_literals

from sklearn import metrics
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

import math
import random
import time

from hazm import *
import numpy as np
import pandas as pd
from collections import Counter
import csv, re, pickle

from colorama import Back, Fore, Style
import time

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import pyclustering as pyclus

import seaborn as sns
sns.set()


# In[2]:


data = pd.read_excel("dataset.xlsx")
reviews = data['question']
# rate = data['Score']
labels = list(data['subject'])


# In[3]:


# stopwords_list(reviews,"dataset1.txt")


# In[4]:


# labels


# In[5]:




RE_USELESS = r'[^\w]'  # remove useless characters
RE_DIGIT = r"^\d+\s|\s\d+\s|\s\d+$"  # remove digits
RE_SPACE = r'\s+'  # remove space
RE_EMAILS = r'[\w\.-]+@[\w\.-]+'
RE_URLS = r'http\S+'
RE_WWW = r'www\S+'


def clean_all_save(document, save_file_path):
    """
    this function generate raw persian text, it remove non-persian character
    and all numbers and symbols
    :param document:
    :param save_file_path:
    :return:
    """
    with open(save_file_path, 'w') as output:
        for sentence in document:
            sentence = clean_sentence(sentence)
            output.write(sentence + '\n')
    return None


def clean_all(document, doc_pattern=r'<TEXT>(.*?)</TEXT>'):
    """
    clean text like hamshahri, irBlogs, and other Treck format
    :param document:
    :param doc_pattern:
    :return:
    """
    clean = ''
    document = re.findall(doc_pattern, document, re.DOTALL)
    for sentence in document:
        sentence = clean_sentence(sentence)
        clean += ' \n' + sentence
    return clean


def clean_sentence(sentence):
    sentence = re.sub(r'[^\u0621-\u06ff]', ' ', sentence)
    sentence = arToPersianChar(sentence)
    sentence = arToPersianNumb(sentence)
    sentence = faToEnglishNumb(sentence)
    sentence = re.sub(r'[a-zA-Z]', ' ', sentence)
    sentence = re.sub(r'[0-9]', ' ', sentence)
    sentence = re.sub(RE_WWW, r' ', sentence)
    sentence = re.sub(RE_URLS, r' ', sentence)
    sentence = re.sub(RE_EMAILS, r' ', sentence)
    sentence = re.sub(RE_USELESS, r' ', sentence)
    sentence = re.sub(RE_DIGIT, r' ', sentence)
    sentence = re.sub(RE_SPACE, r' ', sentence)
    return sentence


def arToPersianNumb(number):
    dic = {
        '١': '۱',
        '٢': '۲',
        '٣': '۳',
        '٤': '۴',
        '٥': '۵',
        '٦': '۶',
        '٧': '۷',
        '٨': '۸',
        '٩': '۹',
        '٠': '۰',
    }
    return multiple_replace(dic, number)


def arToPersianChar(userInput):
    dic = {
        'ك': 'ک',
        'دِ': 'د',
        'بِ': 'ب',
        'زِ': 'ز',
        'ذِ': 'ذ',
        'شِ': 'ش',
        'سِ': 'س',
        'ى': 'ی',
        'ي': 'ی'
    }
    return multiple_replace(dic, userInput)


def faToEnglishNumb(number):
    dic = {
        '۰': '0',
        '۱': '1',
        '۲': '2',
        '۳': '3',
        '۴': '4',
        '۵': '5',
        '۶': '6',
        '۷': '7',
        '۸': '8',
        '۹': '9',
    }
    return multiple_replace(dic, number)


def multiple_replace(dic, text):
    pattern = "|".join(map(re.escape, dic.keys()))
    return re.sub(pattern, lambda m: dic[m.group()], str(text))


# In[6]:


def clean_all(document):
    clean = ''
    for sentence in document:
        sentence = clean_sentence(sentence)
        clean += sentence
    return clean


# In[7]:


j = k = i = 0
reviews1 = []
labels1 = []
# labels1 = list(labels.copy())
normalizer = Normalizer()
for review in reviews:
    sentences = sent_tokenize(normalizer.normalize(clean_all(review)))
    reviews1.extend(sentences)
    for j in range(len(sentences)):
        labels1.insert(i + k, labels[i])
        k += 1
    i += 1


# In[8]:


print(len(reviews1),len(labels1))


# In[9]:


reviews[4]


# In[10]:


#cleaning dataset
words=[]
all_text = ''
# stemmer = Stemmer()
for t in range (len(reviews1)):
    text = reviews1[t]
    text = text.replace('\u200c',' ')
    text = text.replace('\u200f',' ')
    text = re.sub(r'[^a-zA-Z0-9آ-ی۰-۹ ]', ' ', text)
    all_text += text
    all_text += ' '
    wordsInText = text.split()
    for word in wordsInText:
#         word = stemmer.stem(word)
        if word != ' ' or word != '':
            words.append(word)
len(words)


# In[11]:


len(all_text)


# In[12]:


counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

with open("mySavedDict.txt", "wb") as myFile:
    pickle.dump(vocab_to_int, myFile)

'''
with open("mySavedDict.txt", "rb") as myFile:
    myNewPulledInDictionary = pickle.load(myFile)
'''


# In[13]:


# vocab


# In[14]:


reviews_ints = []
for each in reviews1:
    #print (each)
    each = each.replace('\u200c',' ')
    each = each.replace('\u200f',' ')
    each = re.sub(r'[^a-zA-Z0-9آ-ی۰-۹ ]', ' ', each)
    reviews_ints.append([vocab_to_int[word] for word in each.split()])


review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# In[15]:


# reviews_ints[1]


# In[16]:


mi = 15000
su = ma = 0
i = 0
low = []
for each in reviews_ints:
    if len(each) == 2 or len(each) == 1:
        low.append(i)
    if len(each) <= mi:
#         print(each,i,len(each))
        mi = len(each)
    if len(each) > ma:
        ma = len(each)
    su += len(each)
    i += 1
print('min lenght: '+str(mi),' and max lenght: '+str(ma),' and mean lenght: '+str(su/len(reviews_ints)))


# In[17]:


reviews_ints22 = reviews_ints.copy()
for i in range(len(low)):
    print(reviews_ints22.pop(low[len(low)- i -1]),low[len(low)- i -1])
len(reviews_ints22)


# In[18]:


reviews_ints = reviews_ints22.copy()


# In[19]:


# lstm_size = 256
# lstm_layers = 1
# batch_size = 200
# learning_rate = 0.001

# data_dim = 16
# timesteps = 25
# num_classes = 2

n_words = len(vocab)
print (n_words)


# In[54]:


seq_len = 30
features = np.zeros((len(reviews_ints), seq_len), dtype=int)

for i, row in enumerate(reviews_ints):
#     print (i , row)
#     print (i )
#     print ('****')
    features[i, -len(row):] = np.array(row)[:seq_len]
pd.DataFrame(features)


# In[55]:


import configparser
import numpy as np
import pandas as pd

from cluster import Clustering
from genetic import Genetic
from generation import Generation


# In[56]:




NORMALIZATION = True


def readVars(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    budget = int(config.get("vars", "budget"))
    kmax = int(config.get("vars", "kmax"))  # Maximum number of Clusters
    numOfInd = int(config.get("vars", "numOfInd"))  # number of individual
    Ps = float(config.get("vars", "Ps"))
    Pm = float(config.get("vars", "Pm"))
    Pc = float(config.get("vars", "Pc"))

    return budget, kmax, Ps, Pm, Pc, numOfInd


# minmax normalization
def minmax(data):
    normData = data
    data = data.astype(float)
    normData = normData.astype(float)
    for i in range(0, data.shape[1]):
        tmp = data.iloc[:, i]
        # max of each column
        maxElement = np.amax(tmp)
        # min of each column
        minElement = np.amin(tmp)

        # norm_dat.shape[0] : size of row
        for j in range(0, normData.shape[0]):
            normData[i][j] = float(
                data[i][j] - minElement) / (maxElement - minElement)

    normData.to_csv('result/norm_data.csv', index=None, header=None)
    return normData
data = pd.DataFrame(features)
data = minmax(data)  # normalize


# In[30]:



if __name__ == '__main__':
    config_file = "config.txt"
#     if(NORMALIZATION):
#         data = pd.read_csv('data/iris.csv', header=None)

#         data = minmax(data)  # normalize
#     else:
#         data = pd.read_csv('result/norm_data.csv', header=None)

    # size of column
    dim = data.shape[1]

    # kmeans parameters & GA parameters
    generationCount = 0
    budget, kmax, Ps, Pm, Pc, numOfInd = readVars(config_file)

    budget = 30
    kmax = 8
#     numOfInd =20
#     Ps =0.4
#     Pm =0.05
#     Pc =0.8
    
    print("-------------GA Info-------------------")
    print("budget", budget)
    print("kmax", kmax)
    print("numOfInd", numOfInd)
    print("Ps", Ps)
    print("Pm", Pm)
    print("Pc", Pc)
    print("---------------------------------------")

    # dim or pattern id 
    chromosome_length = kmax * dim

    #-------------------------------------------------------#
    # 							main 						#
    #-------------------------------------------------------#
    initial = Generation(numOfInd, 0)
    initial.randomGenerateChromosomes(
        chromosome_length)  # initial generate chromosome

    clustering = Clustering(initial, data, kmax)  # eval fit of chromosomes

    # ------------------calc fitness------------------#
    generation = clustering.calcChromosomesFit()

    # ------------------------GA----------------------#
    while generationCount <= budget:
        GA = Genetic(numOfInd, Ps, Pm, Pc, budget, data, generationCount, kmax)
        generation, generationCount = GA.geneticProcess(
            generation)
        iBest = generation.chromosomes[0]
        clustering.printIBest(iBest)

    # ------------------output result-------------------#
#     clustering.output_result(iBest, data)


# In[23]:


# clustering.getLabels()
a = clustering.getLabels()
len(a)


# In[24]:


GAKMeans_Sil = metrics.silhouette_score(X, a, metric='euclidean')
GAKMeans_Sil


# In[67]:


X = data
colors = np.array(['g', 'r', 'b', 'c', 'k', 'y','royalblue', 'maroon', 'forestgreen',
                   'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy'])


# In[26]:



########## PCA of features for GA_Kmeans
from sklearn.decomposition import PCA
pca_model = PCA(n_components=2)
X_PCA = pca_model.fit_transform(X)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121)
ax.scatter(X_PCA[:, 0], X_PCA[:, 1],c='green', marker='o', s=10)
ax = fig.add_subplot(122)
ax.scatter(X_PCA[:, 0], X_PCA[:, 1], c=colors[a], marker='*')


# In[27]:



#### kmeans algorithm
from sklearn.cluster import KMeans
start = time.time()
kmean = KMeans(n_clusters=8, max_iter=500)
kmean.fit(X)
end = time.time()
print(Fore.BLUE + "k-mean algorithm time is :", end - start)
print(Fore.RESET)



# In[28]:


########## PCA of features for Kmeans
from sklearn.decomposition import PCA
pca_model = PCA(n_components=2)
X_PCA = pca_model.fit_transform(X)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121)
ax.scatter(X_PCA[:, 0], X_PCA[:, 1],c='green', marker='o', s=10)
ax = fig.add_subplot(122)
ax.scatter(X_PCA[:, 0], X_PCA[:, 1], c=colors[kmean.labels_], marker='*')


# In[29]:


from sklearn.cluster import Birch

brc = Birch(branching_factor=50, n_clusters=4, threshold=0.5, compute_labels=True)
brc.fit(X) 
# Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=None,
#    threshold=0.5)
ClusterBirch = brc.predict(X)


# In[30]:


fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121)
ax.scatter(X_PCA[:, 0], X_PCA[:, 1],c='green', marker='o', s=10)
ax = fig.add_subplot(122)
ax.scatter(X_PCA[:, 0], X_PCA[:, 1], c=colors[ClusterBirch], marker='*')


# In[369]:



def WOA_clustering(X, numberOfCluster=3,iterations=100, numberOfWhale=20):
#     X must be numpy.ndarray
    dataPoints=X.shape[0]
    features=X.shape[1]
    #intialise

    centresOfwhale=np.zeros((numberOfWhale,numberOfCluster,features))


    for whale in range(numberOfWhale):
        for cluster in range(numberOfCluster):
            for feature in range(features):
                centresOfwhale[whale,cluster,feature]=float(random.randint(np.min(X[:,feature]),np.max(X[:,feature])))
    bestWhale=0
    for iteration in range(iterations):
    #         print(iteration)
        #dataPointsInCluster=[[[] for cluster in range(numberOfCluster)] for whale in range(numberOfWhale)]
        dataPointsInCluster=np.zeros((numberOfWhale,numberOfCluster))
        bestWhale=0
        bestDist=np.infty
        startTime=time.time()
        for whale in range(numberOfWhale):
            dist=0.00
            clusi = []
            for i in range(numberOfCluster):
                clusi.append(np.zeros((features)))

            for dataPoint in range(dataPoints):
                bestEuclidianDist=np.infty
                bestCluster=0
                for cluster in range(numberOfCluster):
                    euclidDist=np.linalg.norm(centresOfwhale[whale,cluster]-X[dataPoint,:])
                    if(euclidDist<bestEuclidianDist):
                        bestEuclidianDist=euclidDist
                        bestCluster=cluster
                dist=dist+bestEuclidianDist
                #dataPointsInCluster[whale][bestCluster].append(dataPoint)
                dataPointsInCluster[whale][bestCluster]=dataPointsInCluster[whale][bestCluster]+1

                for i in range(numberOfCluster):
                    if bestCluster==i:
                        clusi[i]=clusi[i]+X[dataPoint]

            if(dist<bestDist):
                bestDist=dist
                bestWhale=whale
            for i in range(numberOfCluster):
                if(dataPointsInCluster[whale][i]!=0):
                    centresOfwhale[whale][i]=clusi[i]/dataPointsInCluster[whale][i]
#         print(time.time()-startTime)
        #shift the centroid in the centre of the datapoints inside cluster

   
    
        startTime=time.time()
#         print("shift started")
        for whale in range(numberOfWhale):
            for cluster in range(numberOfCluster):
#                 print(dataPointsInCluster)
                numberOfPoint=dataPointsInCluster[whale][cluster]
#                 print(numberOfPoint)
                if numberOfPoint==0 :
                    continue
                for feature in range(features):
                    sum=0.00
#                     for dataPoint in dataPointsInCluster[whale][cluster]:
                    sum=sum+X[int(dataPointsInCluster[whale][cluster]-1)][feature]
                    sum=sum/numberOfPoint
                    centresOfwhale[whale,cluster,feature]=sum

        #shift centroids using equations of WOA
    #     print(time.time()-startTime)
        startTime=time.time()
    #     print("woa started")
        a=2-iteration*((2.00)/iterations) #eqn 2.3
        a2=-1+iteration*((-1.00)/iterations)
        for whale in range(numberOfWhale):
            r1=random.random()
            r2=random.random()
            A=2*a*r1-a  # Eq. (2.3) in the paper
            C=2*r2      # Eq. (2.4) in the paper
            p=random.random()
            b=1                          #  parameters in Eq. (2.5)
            l=(a2-1)*random.random()+1   #  parameters in Eq. (2.5)

            for cluster in range(numberOfCluster):
                if p<0.5 :
                    if abs(A)>=1 :
                        rand_leader_index = int(math.floor((numberOfWhale-1)*random.random()+1));
                        X_rand = centresOfwhale[rand_leader_index]
                        D_X_rand=abs(C*X_rand[cluster]-centresOfwhale[whale,cluster]) # Eq. (2.7)
                        centresOfwhale[whale,cluster]=X_rand[cluster]-A*D_X_rand      # Eq. (2.8)
                    elif abs(A)<1 :
                        D_Leader=abs(C*centresOfwhale[bestWhale,cluster]-centresOfwhale[whale,cluster]) # Eq. (2.1)
                        centresOfwhale[whale,cluster]=centresOfwhale[bestWhale,cluster]-A*D_Leader      # Eq. (2.2)
                elif p>=0.5 :
                    distance2Leader=abs(centresOfwhale[bestWhale,cluster]-centresOfwhale[whale,cluster])      # Eq. (2.5)
                    centresOfwhale[whale,cluster]=distance2Leader*math.exp(b*l)*math.cos(l*2*3.14)+centresOfwhale[bestWhale,cluster]
    #     print(time.time()-startTime)
        startTime=time.time()
    mins = []
    for i in range(numberOfCluster):
        mins.append([np.infty])
    WOACluster=[0 for dataPoint in range(dataPoints)]
    j = 0
    for dataPoint in range(dataPoints):
        d = []
        for i in range(numberOfCluster):
            d.append(np.linalg.norm(centresOfwhale[bestWhale,i]-X[dataPoint,:]))
            if d[i] <= mins[i][0]:
                mins[i][0] = d[i]
                mins[i].append(j)
        j += 1
        WOACluster[dataPoint] = d.index(min(d))
    j = 0
    for i in mins:
#         print(i[1:],j)
        WOACluster = pd.DataFrame(WOACluster)
        WOACluster.iloc[i[1:]] = j
        WOACluster = list(WOACluster[0])
        j += 1

    return WOACluster


# In[396]:


# centresOfwhale=[]
# bestWhale = 0
sill = -10
for i in range(25):
    labeles = WOA_clustering(X.values, numberOfCluster=8,iterations=100, numberOfWhale=30)
    try:
        sill1 = metrics.silhouette_score(X.values, labeles, metric='euclidean')
        if sill1> sill:
            sill =sill1
            WOACluster = labeles
    except:
        1+1

# print(WOACluster)
# centresOfwhale


# In[397]:


WOAKMeans_Sil = metrics.silhouette_score(X.values, WOACluster, metric='euclidean')
WOAKMeans_Sil


# In[398]:


fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121)
ax.scatter(X_PCA[:, 0], X_PCA[:, 1],c='green', marker='o', s=10)
ax = fig.add_subplot(122)
ax.scatter(X_PCA[:, 0], X_PCA[:, 1], c=colors[WOACluster], marker='*')


# In[34]:


GAKMeans_Sil = metrics.silhouette_score(X, a, metric='euclidean')
GAKMeans_Sil


# In[35]:


labels = kmean.labels_
KMeans_Sil = metrics.silhouette_score(X, kmean.labels_, metric='euclidean')
print('Kmeans silhouette ',KMeans_Sil)


# In[36]:



print("Birch Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, ClusterBirch, metric='sqeuclidean'))


# In[37]:


WOAKMeans_Sil = metrics.silhouette_score(X, WOACluster, metric='euclidean')
WOAKMeans_Sil


# In[ ]:





# In[309]:


# normalizer = Normalizer()
# normalizer.normalize('اصلاح نويسه ها و استفاده از نیم‌فاصله \n پردازش را آسان مي كند')
# # 'اصلاح نویسه‌ها و استفاده از نیم‌فاصله پردازش را آسان می‌کند'


# In[310]:


# sent_tokenize('ما هم برای وصل کردن آمدیم! ولی برای پردازش، جدا بهتر نیست؟')
# # ['ما هم برای وصل کردن آمدیم!', 'ولی برای پردازش، جدا بهتر نیست؟']


# In[107]:


# word_tokenize('ولی برای پردازش، جدا بهتر نیست؟')
# # ['ولی', 'برای', 'پردازش', '،', 'جدا', 'بهتر', 'نیست', '؟']


# In[108]:


# stemmer = Stemmer()
# print(stemmer.stem(stemmer.stem('پردازش‌ها')))
# # 'کتاب'


# In[109]:


# lemmatizer = Lemmatizer()
# lemmatizer.lemmatize('می‌روم')
# 'رفت#رو'


# In[110]:


# tagger = POSTagger(model='resources/postagger.model')
# tagger.tag(word_tokenize('ما بسیار کتاب می‌خوانیم'))
# [('ما', 'PRO'), ('بسیار', 'ADV'), ('کتاب', 'N'), ('می‌خوانیم', 'V')]


# In[111]:


# chunker = Chunker(model='resources/chunker.model')
# tagged = tagger.tag(word_tokenize('کتاب خواندن را دوست داریم'))
# tree2brackets(chunker.parse(tagged))
# '[کتاب خواندن NP] [را POSTP] [دوست داریم VP]'


# In[34]:


# parser = DependencyParser(tagger=tagger, lemmatizer=lemmatizer)
# parser.parse(word_tokenize('زنگ‌ها برای که به صدا درمی‌آید؟'))


# In[ ]:




