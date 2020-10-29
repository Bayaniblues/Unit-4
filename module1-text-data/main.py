import re
from collections import Counter
import pandas as pd
import seaborn as sns
import squarify
import matplotlib.pyplot as plt
from collections import Counter
import spacy
from spacy.tokenizer import Tokenizer

from collections import Counter


class HandleTokens(object):
    """
    Created these functions as modules, some important methods used to finish assignment 1.
    """
    @staticmethod
    def tokenize(df_in):
        """
        Tokenize by inputting a dataframe.
        :param df_in:
        :return:
        """
        nlp = spacy.load("en_core_web_lg")

        # Tokenizer
        tokenizer = Tokenizer(nlp.vocab)
        tokens = []
        for doc in tokenizer.pipe(df_in, batch_size=500):
            doc_tokens = [token.text for token in doc]
            tokens.append(doc_tokens)
        return tokens

    @staticmethod
    def count(docs):
        """
        Count words in
        :param docs:
        :return:
        """
        word_counts = Counter()
        appears_in = Counter()

        total_docs = len(docs)

        for doc in docs:
            word_counts.update(doc)
            appears_in.update(set(doc))

        temp = zip(word_counts.keys(), word_counts.values())

        wc = pd.DataFrame(temp, columns=['word', 'count'])

        wc['rank'] = wc['count'].rank(method='first', ascending=False)
        total = wc['count'].sum()

        wc['pct_total'] = wc['count'].apply(lambda x: x / total)

        wc = wc.sort_values(by='rank')
        wc['cul_pct_total'] = wc['pct_total'].cumsum()

        t2 = zip(appears_in.keys(), appears_in.values())
        ac = pd.DataFrame(t2, columns=['word', 'appears_in'])
        wc = ac.merge(wc, on='word')

        wc['appears_in_pct'] = wc['appears_in'].apply(lambda x: x / total_docs)

        return wc.sort_values(by='rank')

    @staticmethod
    def stopwords(list_in):
        nlp = spacy.load("en_core_web_lg")

        # Tokenizer
        tokenizer = Tokenizer(nlp.vocab)
        return nlp.Defaults.stop_words.union(list_in)

    @staticmethod
    def combine_stopwords(dataframe_in, stopword_dict):
        """
        Please use the above function, stopwords.
        :param dataframe_in:
        :param stopword_dict:
        :return:
        """
        nlp = spacy.load("en_core_web_lg")

        # Tokenizer
        tokenizer = Tokenizer(nlp.vocab)

        tokens = []

        for doc in tokenizer.pipe(dataframe_in, batch_size=500):

            doc_tokens = []

            for token in doc:
                if token.text.lower() not in stopword_dict:
                    doc_tokens.append(token.text.lower())

            tokens.append(doc_tokens)

        return tokens

    @staticmethod
    def lemma_attributes(doc):
        """
        Input doc to print lemma attributes
        :param doc:
        :return:
        """
        for token in doc:
            print(token.text, "  ", token.lemma_)


    @staticmethod
    def get_lemmas(text):
        nlp = spacy.load("en_core_web_lg")

        lemmas = []

        doc = nlp(text)

        # Something goes here :P
        for token in doc:
            if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON'):
                lemmas.append(token.lemma_)

        return lemmas


class FruitfulFunctions(object):
    """
    Notes from lecture
    """
    @staticmethod
    def df_token(input_df):
        """
        Tokenization, input string type to output list.
        :param input_df:
        :return:
        """
        return list(input_df)

    @staticmethod
    def df_counts(input_df, bool=True, range_it=50):
        return input_df.value_counts(normalize=bool)[:range_it]

    @staticmethod
    def df_split(input_df):
        """

        :param input_df:
        :return:
        """
        return input_df.split(" ")

    @staticmethod
    def df_lower(input_df):
        """
        Case Normalization, input df['example'], to output values as lowercase
        :param input_df:
        :return:
        """
        return input_df.apply(lambda x: x.lower())

    @staticmethod
    def df_upper(input_df):
        """
        Case Normalization, input df['example'], to output values as uppercase
        :param input_df:
        :return:
        """
        return input_df.apply(lambda x: x.upper())

    @staticmethod
    def df_alphanumeric(input_df):
        """

        :param input_df:
        :return:
        """
        return re.sub('[^a-zA-Z 0-9]', '', input_df)

    @staticmethod
    def df_raw_count(input_df):
        """

        :param input_df:
        :return:
        """
        return input_df.value_counts(normalize=True)[:50]

    @staticmethod
    def count_tokens(df_in, integer=10):
        """
        counts words
        :param df_in:
        :param integer:
        :return:
        """

        word_counts = Counter()
        df_in.apply(lambda x: word_counts.update(x))
        return word_counts.most_common(integer)

    @staticmethod
    def count_pipeline(df_in):
        """
        Takes a corpus of document and returns a dataframe of word counts to analyze
        :param df_in:
        :return:
        """
        word_counts = Counter()
        appears_in = Counter()

        total_docs = len(df_in)

        for doc in df_in:
            word_counts.update(doc)
            appears_in.update(set(doc))

        temp = zip(word_counts.keys(), word_counts.values())

        wc = pd.DataFrame(temp, columns=['word', 'count'])

        wc['rank'] = wc['count'].rank(method='first', ascending=False)
        total = wc['count'].sum()

        wc['pct_total'] = wc['count'].apply(lambda x: x / total)

        wc = wc.sort_values(by='rank')
        wc['cul_pct_total'] = wc['pct_total'].cumsum()

        t2 = zip(appears_in.keys(), appears_in.values())
        ac = pd.DataFrame(t2, columns=['word', 'appears_in'])
        wc = ac.merge(wc, on='word')

        wc['appears_in_pct'] = wc['appears_in'].apply(lambda x: x / total_docs)

        return wc.sort_values(by='rank')


class Visualize(object):

    @staticmethod
    def distribution_plot(df_in):
        """
        inputs dataframe and returns seaborn lineplot
        :param df_in:
        :return:
        """
        return sns.lineplot(x='rank', y='cul_pct_total', data=df_in)

    @staticmethod
    def square_plot(wc):
        """

        :param wc:
        :return:
        """
        wc_top20 = wc[wc['rank'] <= 20]

        squarify.plot(sizes=wc_top20['pct_total'], label=wc_top20['word'], alpha=.8)
        plt.axis('off')
        return plt.show()

    @staticmethod
    def trimming(wc):
        return sns.lineplot(x='rank', y='cul_pct_total', data=wc);


class SpacyFruitfulFunctions(object):
    @staticmethod
    def stem_this(list_in):
        """
        stemming removes last few letters of a words
        :param list_in:
        :return:
        """
        ps = PorterStemmer()
        state = []
        for word in list_in:
            state.append(ps.stem(word))
        return state


if __name__ == "__main__":
    print("test here")
