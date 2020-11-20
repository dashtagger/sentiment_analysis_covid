from wordcloud import WordCloud, ImageColorGenerator
from datapipeline.clean_data import clean_pipeline
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import pandas as pd
import pickle


class Wordcloud_gen():
    def __init__(self, datapath, uniquewords=None, min_length=33):
        self.unique_word_path = '../data/unique_word.txt' # Path to the unique
                                                          # word list in pickle
                                                          # file
        if uniquewords is not None:
            self.unique_word_path = uniquewords
        self.df = pd.read_csv(datapath, index_col=0)
        self.df = clean_pipeline(self.df, min_length)


    def pipeline(self, savepath, remove_common=True):
        """ Main pipeline for this class.
            After initialising, just run this pipeline to get the word cloud
            figure save to the savepath.

        Args:
            savepath (string): the path to save the word cloud figure
            remove_common (bool, optional): To remove common words from the
                corpus. Defaults to True.
        """
        self.get_corpus()
        self.join_corpus()
        if remove_common:
            self.remove_common_words()
        self.wordcloud(savepath)


    def get_corpus(self, df=None):
        """ Extract list of tweets from the dataframe

        Args:
            df (pandas df, optional): df of tweets sentiment. Defaults to None.

        Returns:
            corpus: list of tweets
        """
        if df is None:
            df = self.df
        self.corpus = df.tweet.values
        return self.corpus


    def join_corpus(self, corpus=None):
        """ To combine a list of tweets into one large corpus

        Args:
            corpus (list, optional): list of tweets. Defaults to None.

        Returns:
            joincorpus (string): Corpus of all tweets join together
                into one element.
        """
        if corpus is None:
            corpus = self.corpus
        self.joincorpus = ' '.join(corpus)
        return self.joincorpus


    def remove_common_words(self, joincorpus=None):
        """ Remove common words from the corpus

        Args:
            joincorpus (string, optional): Corpus of all tweets join together
                into one element. Defaults to None.

        Returns:
            joincorpus: joincorpus with common words removed
        """
        if joincorpus is None:
            joincorpus = self.joincorpus
        with open(self.unique_word_path, 'rb') as f:
            unique_words = pickle.load(f)
        w_tokens = [w for w in word_tokenize(joincorpus)
                    if w in unique_words]
        self.joincorpus = ' '.join(w_tokens)
        return self.joincorpus


    def wordcloud(self,
                  savepath,
                  joincorpus=None,
                  max_words=50,
                  max_font_size=50):
        """ Generate wordcloud and save the figure to the savepath

        Args:
            savepath (string): the path to save the word cloud figure
            joincorpus ([type], optional): Corpus of all tweets join together
                into one element. Defaults to None.
            max_words (int, optional): max word in word cloud. Defaults to 50.
            max_font_size (int, optional): max font size of word cloud.
                Defaults to 50.
        """
        if joincorpus is None:
            joincorpus = self.joincorpus
        wc = WordCloud(max_font_size=max_font_size,
                       max_words=max_words,
                       width=800,
                       height=400,
                       collocations=False)
        wc = wc.generate(joincorpus)
        plt.imshow(wc, interpolation='bilinear')
        plt.title('Wordcloud of dataset')
        plt.axis('off')
        plt.savefig(savepath)
