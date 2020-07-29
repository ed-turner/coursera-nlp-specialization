from typing import List, Callable, Union
from functools import reduce, partial
import pandas as pd

import nltk  # Python library for NLP
from nltk.corpus import twitter_samples  # sample Twitter dataset from NLTK
import plotly.express as px  # library for visualization
import random  # pseudo-random number generator

import re  # library for regular expression operations
import string  # for string operations

from nltk.corpus import stopwords  # module for stop words that come with NLTK
from nltk.stem import PorterStemmer  # module for stemming
from nltk.tokenize import TweetTokenizer  # module for tokenizing strings


def init():
    """
    This only is required if we need to init the process
    :return:
    """
    nltk.download('twitter_samples')


def get_twitter_data() -> pd.DataFrame:
    """
    This will return a pandas.DataFrame with the labelled twitter data
    :return:
    """
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    df1 = pd.DataFrame(data=all_positive_tweets, columns=['tweets'])
    df2 = pd.DataFrame(data=all_negative_tweets, columns=['tweets'])

    df1["label"] = 1
    df2["label"] = 0

    return pd.concat([df1, df2], ignore_index=True)


def regex_processer(tweet: str) -> str:
    """

    :param tweet:
    :return:
    """

    funct_lst = [
        partial(re.sub, pattern=r'^RT[\s]+', repl=''),
        partial(re.sub, pattern=r'https?:\/\/.*[\r\n]*', repl=''),
        partial(re.sub, pattern=r'#', repl='')
    ]
    return reduce(lambda text, funct: funct(text), funct_lst, tweet)


def process_tweets(tweets_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param tweets_df:
    :return:
    """

    stopwords_english = stopwords.words('english')

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

    stemmer = PorterStemmer()

    def filter_funct(tweet_tokens):
        return list(
            filter(
                lambda word: (word not in stopwords_english) and (word not in string.punctuation),
                tweet_tokens)
        )

    def stem_funct(tweet_tokens: List[str]) -> List[str]:
        return list(map(stemmer.stem, tweet_tokens))

    def process_funct(x: str) -> List[str]:
        return stem_funct(
            filter_funct(
                tokenizer.tokenize(
                    regex_processer(x)
                )
            )
        )

    return tweets_df.assign(processed_tweets=tweets_df["tweets"].apply(process_funct))
