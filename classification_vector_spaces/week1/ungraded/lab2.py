from typing import Dict

import plotly.express as px  # library for visualization
import pandas as pd

from utils.stats import count_freq_across_documents


def sentiment_word_graph(docs_df: pd.DataFrame, output_path: str, doc_col: str = "doc") -> None:
    """
    This will create a word graph based on the counts per word in the docs

    :param docs_df: The data frame with the documents
    :param output_path: The output path
    :param doc_col: The column with our documents
    :return:
    """

    sentiment_counts: pd.DataFrame = \
        docs_df.groupby("label").apply(lambda x: pd.DataFrame.from_dict(
            data=x[doc_col].apply(count_freq_across_documents),
            columns=["n"],
            orient="index"
        ).reset_index().rename(columns={"index": "word"})).reset_index()

    sentiment_counts['log_counts'] = sentiment_counts['count'].log()

    pivoted_sentiments = pd.pivot(sentiment_counts, index=["word"], values="log_counts", columns=["label"])

    fig = px.scatter(data_frame=pivoted_sentiments, x='positive', y='negative', text='word')

    fig.to_html(f"{output_path}/word_graph.html")

