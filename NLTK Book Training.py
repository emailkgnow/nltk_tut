__author__ = 'khalid'

# Alt + F12 moves between Editor and Terminal

# This training from the NLTK bookj @ http://www.nltk.org/book/

################## CHAPTER 1 ####################
from nltk.book import *
import nltk

print()

text8.concordance('love') # The context the word is used in

text8.similar('love') # Returns the words that are used in a similar context to love

len(text8) # Number of words in a text

set(text8) # Returns all the distinct words in a given text

len(set(text8)) # Number of distinct words in a text

len(set(text8))/ len(text8) # Shows that distinct words are only 6% of total words in text3; text1 is 13.6% which means it is richer

text8.dispersion_plot(['love', 'hat','man']) # Returns how the words are dispersed in the text

text8.common_contexts(["monstrous", "very"]) # Returns the common contexts. note the brackets.

text8.count('man') # returns how many time a words occurs in a text

def word_counter (text):
    """Returns a Dict of all the distinct words as keys and how many
    times they appear in a text as values"""

    word_count = {}
    for word in set(text):
        word_count[word] = text.count(word)
    return word_count

fd = nltk.FreqDist(text8) # Returns frequency distribution object with all words sorted decending with tally

fd.plot(50,cumulative = True) # Plots the 50 most common cumulatively (leave it out to see reg. plot)

V = set(text8)
long_words = [w for w in V if len(w) > 15] # Returns list of words longer than 15 letters

x = sorted(w for w in set(text8) if len(w) > 7 and fd[w] > 7) # returns a list of words with more than 7 words and that appear mores than 7 times. Note: sorted() automatically returns a list so you don't have to call list()

text8.collocations() # Returns collections (words that appear together)

s='Faith'
t='F'
s.startswith(t)	# test if s starts with t
s.endswith(t)	# test if s ends with t
t in s	# test if t is a substring of s
s.islower()	#test if s contains cased characters and all are lowercase
s.isupper()	#test if s contains cased characters and all are uppercase
s.isalpha()	#test if s is non-empty and all characters in s are alphabetic
s.isalnum()	#test if s is non-empty and all characters in s are alphanumeric
s.isdigit()	#test if s is non-empty and all characters in s are digits
s.istitle()	#test if s contains cased characters and is titlecased (i.e. all words in s have initial capitals)

nltk.chat.chatbots() # returns different chatbots you can converse with


################## CHAPTER 2 ####################

from nltk.corpus import gutenberg # imports some samples from the gutenberg project

gutenberg.fileids()

emma = gutenberg.words('austen-emma.txt')
