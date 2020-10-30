from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import spacy
nlp = spacy.load("en_core_web_lg")

class Notes(object):
    """
    Some important notes from lecture.
    """

    @staticmethod
    def test():
        return 'test'

    @staticmethod
    def tokenize(document):
        """
        input document

        :returns a tokenized lemma list
        """
        doc = nlp(document)
        return [token.lemma_ for token in doc if (token.is_stop != True) and (token.is_punct != True)]

    @staticmethod
    def gather_data(filefolder):
        """ Produces List of Documents from a Directory

        filefolder (str): a path of .txt files

        :returns list of strings
        """

        data = []

        files = os.listdir(filefolder)  # Causes variation across machines

        for article in files:

            path = os.path.join(filefolder, article)

            if path[-3:] == 'txt':  # os ~endswith('txt')
                with open(path, 'rb') as f:
                    data.append(f.read())

        return data

    @staticmethod
    def vectorize(text2fit, test2tranform):
        """
        input text to fit and text to transform

        :returns a vectorized dtm
        """
        from sklearn.feature_extraction.text import CountVectorizer
        vect = CountVectorizer()
        vect.fit(text2fit)
        dtm = vect.transform(test2tranform)
        return dtm

    @staticmethod
    def dtm_word_count(vectorize, vect, text_in):
        """
        inputs vectorize model and vect for column,

        :return: word count.
        """
        dtm = vectorize(text_in)
        output = pd.DataFrame(dtm.todense(), columns=vect.get_feature_names())
        return output

    @staticmethod
    def distribution_plot(data):
        """
        Inputs dataframe to visualize our distribution
        """
        doc_len = [len(doc) for doc in data]
        import seaborn as sns

        return sns.distplot(doc_len)

    @staticmethod
    def Term_frequency(data):
        """
        Percentage of words in a document

        Document Frequency: A penalty for the word existing in a high number of documents.

        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Instantiate vectorizer object
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

        # Create a vocabulary and get word counts per document
        # Similiar to fit_predict
        dtm = tfidf.fit_transform(data)

        # Print word counts

        # Get feature names to use as dataframe column headers
        dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())

        return dtm

    @staticmethod
    def cosine_sim(dtm):
        """
        import dtm
        Calculate Distance of TF-IDF Vectors

        """
        from sklearn.metrics.pairwise import cosine_similarity

        dist_matrix = cosine_similarity(dtm)
        df = pd.DataFrame(dist_matrix)
        return df


class KnnState(object):
    from sklearn.neighbors import NearestNeighbors
    """
    Used to save our NearestNeighbors model
    """
    def __init__(self):
        self._nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')

    @property
    def test(self):
        return self._nn

    @test.setter
    def test(self, value):
        self._nn = value


class Knn(object):
    @staticmethod
    def fit(nn, dtm):
        """fits our model"""
        return nn.fit(dtm)

    @staticmethod
    def values(nn, dtm, row=0):
        """shows values of our model"""
        return nn.kneighbors([dtm.iloc[row].values])

    @staticmethod
    def query(nn, dtm, row=256):
        """Query a specific row from a dtm"""
        return nn.kneighbors([dtm.iloc[row]])

    @staticmethod
    def density(nn, new):
        """Check density"""
        nn.kneighbors(new.todense())


class Word2Vec(object):
    from sklearn.decomposition import PCA
    @staticmethod
    def similarity(a, b):
        """
        Input two nlp() classes to determine similarity

        """
        similarity = a.similarity(b)
        return similarity

    @staticmethod
    def get_word_vectors(words):
        """
        converts a list of words into their word vectors
        """
        return [nlp(word).vector for word in words]

    @staticmethod
    def process_PCA(get_word_vectors):
        """
        intialise pca model and tell it to project data down onto 2 dimensions

        fit the pca model to our 300D data, this will work out which is the best
        way to project the data down that will best maintain the relative distances
        between data points. It will store these intructioons on how to transform the data.

        Tell our (fitted) pca model to transform our 300D data down onto 2D using the
        instructions it learnt during the fit phase.

        let's look at our new 2D word vectors
        """
        words = ['car', 'truck', 'suv', 'race', 'elves', 'dragon', 'sword', 'king', 'queen', 'prince', 'horse', 'fish',
                 'lion', 'tiger', 'lynx', 'potato']

        # intialise pca model and tell it to project data down onto 2 dimensions
        pca = PCA(n_components=2)

        # fit the pca model to our 300D data, this will work out which is the best
        # way to project the data down that will best maintain the relative distances
        # between data points. It will store these intructioons on how to transform the data.
        pca.fit(get_word_vectors(words))

        # Tell our (fitted) pca model to transform our 300D data down onto 2D using the
        # instructions it learnt during the fit phase.
        word_vecs_2d = pca.transform(get_word_vectors(words))

        # let's look at our new 2D word vectors
        return word_vecs_2d

