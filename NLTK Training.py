###########     Natural Language Toolkit (nltk) ##############

#this is training from this youtube playlist
#https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL


import nltk

#this will download all the required corpuses and other files
#nltk.download()


from nltk.tokenize import word_tokenize #this will give you the ability to tokenise (separate) the words
from nltk.corpus import stopwords #this will give you a list of all the words that don’t usually carry meaning
from nltk.corpus import state_union
#from nltk.corpus import PunktSentenceTokenizer
from nltk.stem import PorterStemmer     #this give the stem of the word to help “normalize’ text
from nltk.stem import WordNetLemmatizer #this is like stemming, but gives a complete word or synonym
from nltk.corpus import wordnet, movie_reviews #movie_reviews are 1000 positive and 1000 negative movie reviews
import random #this is to randomize the movie reviews as the first 1000 are positive and the other 1000 negative
import pickle





my_text = """The World Wide Web, or simply Web, is a way of accessing information over the medium of the Internet. It is an information-sharing model that is built on top of the Internet. The Web uses the HTTP protocol, only one of the languages spoken over the Internet, to transmit data. Web services, which use HTTP to allow applications to communicate in order to exchange business logic, use the the Web to share information. The Web also utilizes browsers, such as Internet Explorer or Firefox, to access Web documents called Web pages that are linked to each other via hyperlinks. Web documents also contain graphics, sounds, text and video.
The Web is just one of the ways that information can be disseminated over the Internet. The Internet, not the Web, is also used for e-mail, which relies on SMTP, Usenet news groups, instant messaging and FTP. So the Web is just a portion of the Internet, albeit a large portion, but the two terms are not synonymous and should not be confused."""

address = state_union.raw('2006-GWBush.txt')


def stem_text (text):
    """reduces the text to its stems and removes the stop words"""
    tokenized_text = word_tokenize(text)
    #this is a list comp that filters the stopwords from  tokenized text
    stopped_text = [word for word in tokenized_text if word not in stopwords.words('english')] #note english in stopwords
    stemmed_list =[]
    #this give the stem of the word to help “normalize’ text
    ps = PorterStemmer()
    for word in stopped_text:
        x = ps.stem(word)
        stemmed_list.append(x)
    print('text has been stemmed')
    return stemmed_list

def tagged_text (text):
    """tags a sequence with part of speech"""
    #tokenized_text = word_tokenize(text)
    #stopped_text = [word for word in tokenized_text if word not in stopwords.words('english')]
    stemmed = stem_text(text)
    tagged=nltk.pos_tag(stemmed) #this tags a sequence also makes use of a previous func

    print('text has been tagged by tagged_text function')
    return tagged


def chunking (text):
    """this seems to group words of close proximity to eachother or "chunk words" to better exract meaning from the text"""
    tagged = tagged_text(text) #make use of a previous func
    chunkGram=r"Chunk: {<RB.?>*<VB.?>*<NNP><NN>?}"
    chinkGram=r"Chink: {<.*>+}}<VB.?|IN|DT>+{" #chinking is the opposite of chunking, but this pattern is working!!
    chunkParser=nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    print ('text has been chunked')
    print (chunked)

def name_entity (text):
    """this tries to infer PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, FACILITY, GPE (GEOGRAPHICAL LOCATION)"""
    tagged = tagged_text(text) #make use of a previous func
    namedEnt = nltk.ne_chunk(tagged, binary=True) #here is the magic, note binary just tells if its a named entity or not
    print(namedEnt)

def lemmatize_text (text):
    """reduces the text to real word (not stem) or gives a synonym"""
    tokenized_text = word_tokenize(text)
    #this is a list comp that filters the stopwords from  tokenized text
    stopped_text = [word for word in tokenized_text if word not in stopwords.words('english')]
    lemmatized_list =[]
    #this give the stem of the word to help “normalize’ text
    lm = WordNetLemmatizer()
    for word in stopped_text:
        x = lm.lemmatize(word)
        lemmatized_list.append(x)
    print('text has been lemmatized')
    print(lemmatized_list)
    return lemmatized_list

def word_finder (word):
    """utilize wordnet to find synonymos"""
    syn = wordnet.synsets(word)
    print('this is the .name():')
    print(syn[0].name(),'\n')
    print('this is the .lemmas()[0].name():')
    print(syn[0].lemmas()[0].name(),'\n')
    print('this is the .definition():')
    print(syn[0].definition(),'\n')
    print('this is the .example():')
    print(syn[0].examples(),'\n')


def synonym_def_generator(word):
    """finds definitions of synonyms of a given word + example"""
    synonyms = wordnet.synsets(word)
    for i in range(len(synonyms)):
        print(synonyms[i].name(), '- ',synonyms[i].definition(), '\n - EXAMPLE :-', synonyms[i].examples(),'\n')

def def_gen(word):
    pass

def find_similarity (word1,word2):
    """returns a decimel indicating similarity between 2 words"""
    syn1 = wordnet.synsets(word1)
    syn2 = wordnet.synsets(word2)


    w1=syn1[0].name() #to get word entry like ship.n.01
    w2=syn2[0].name()

    syn_set_w1=wordnet.synset(w1)
    syn_set_w2=wordnet.synset(w2)

    print(syn_set_w1.wup_similarity(syn_set_w2)*100) # the *100 to turn into percentage (maybe wrong??)

def analyze_movie_reviews(document):
    """this func will the NIAVE BAYES algorithm to identify words that are associated with positive or negative
     reviews (based on pos or neg previously assigned review categories in a given set of movie reviews) and use that to cast judgement
     on other reviews that have not been categorized"""


    all_words=[] # this will contain all the words of the 2000 reviews which is 1,583,820 words via the loop below
    for w in movie_reviews.words():
        all_words.append(w.lower()) # we use .lower() to normalize data

    all_words = nltk.FreqDist(all_words) # all_words is converted to frequency distrip object that contains all the words and how many times they occur odered from most to least occuring words
    #print(all_words.most_common(10)) # the new object is passed  n and returns n most frequent words
    word_features = list(all_words.keys())[:3000] # will give the top 3000 words (features) that we will test against.
                                                  # he says we will see which words are most common in pos and neg... (so he said)

    words = set(document) # removes duplicate words from the argument of the function wich is one pos or neg movie review

    features = {} # dict to carry what features are present in a given documents the key will be the word and a boolean will be the value
    for word in word_features:
        features[word] = (word in words) # (word in words) will return a True or False

    return features # a dict of the top 3000 words in all the reviews with the word as key and boolean ans value. the word is in
                    # the review it will be true..etc
                    # SINCE THE MOVIE IS CATEGORIZED AS POS OR NEG WE WILL KNOW THAT A GIVEN WORD IS TRUE IN POS OR NEG REVIEW
                    # LATER WE CAN USE THAT TO ANALYZE UNCATEGORIZED REVIEWS

# the below list comp creates tuple of: a list of all the words of one moview review and pos or neg judgement of a movie
# review. you will analyze the words associated with pos or neg reviews and use that to classify other review as pos or neg
documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)] #note that category and movie_review.categories is either pos or neg and fileid is the file name of the reivew
random.shuffle(documents) #just to shuffle between pos and neg reviews as are ordered by category


#lemmatize_text(my_text)
#word_finder('plan')
#synonym_def_generator('think')
#find_similarity('good','bad')
print((analyze_movie_reviews(movie_reviews.words('neg/cv000_29416.txt'))))


featuresets=[(analyze_movie_reviews(rev),category) for (rev, category) in documents]
# a list with boolean vlues of top 3000 words + pos or neg category assigned so i will be able to boolean values to of
# neg or pos categories to words so later I can analyze uncategorized reviews.... long story.

training_set = featuresets[:1900] #used to train
testing_set = featuresets[1900:]  #used to check if training was good

classifier = nltk.NaiveBayesClassifier.train(training_set) #this is how we train "classifier" is usied on testing set below
print("Naive Bayes Accurcy Percentage:", (nltk.classify.accuracy(classifier,testing_set))*100) # this is how we test. returns accuracy percentage
classifier.show_most_informative_features(15) #this will show the most clearly distinguishing feature

#this is how pickle an object to reduce time create a "jar" (my words), insert file in jar, close jar. opposite to reopen
save_classifier = open('naivebayes.pickle','wb') # to pickle, you have to create a "pickle jar" to put in your file. note 'wb' writing bytes
pickle.dump(classifier,save_classifier) # here I'm pickling the file in the "pickle jar"
save_classifier.close() #gotta close that jar, don't ya..

#this is to how to load a pickled object
# open_classifier = open('naivebayes.pickle','rb') #opening the 'pickle jar' note the 'rb' ie read bytes
# classifier = pickle.load(open_classifier) #removing the pickles from the 'pickle jar'
# open_classifier.close()
