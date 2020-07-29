from typing import List, Dict, Iterable, Callable
from functools import reduce, partial


def combine_count_dict(d1: Dict[str, int], d2: Dict[str, int]) -> Dict[str, int]:
    """
    This combines both the count dictionaries

    :param d1: The left count dictionary
    :param d2: The right count dictionary
    :return:
    """

    d1_keys = list(d1.keys())
    d2_keys = list(d2.keys())

    all_keys = list(set(d1_keys) | set(d2_keys))

    return {key: d1.get(key, 0) + d2.get(key, 0) for key in all_keys}


def count_freq_across_text(text: str, delimiter: str) -> Dict[str, int]:
    """
    This takes a text, and returns a dictionary that will have all of the counts
    per delimiter

    :param text: The text to count text by
    :param delimiter: The delimiter to split the text by.  The default is the " "
    :return:
    """
    # we split the data according the the delimiter
    text_lst: List[str] = text.split(delimiter)

    return reduce(combine_count_dict, [{x: 1} for x in text_lst], dict())


def count_freq_across_documents(docs: Iterable[str], delimiter: str = " ") -> Dict[str, int]:
    """
    This takes an iterable of all documents, and returns a dictionary that will have all of the counts
    per delimiter

    :param docs: The iterable of docs
    :param delimiter: The delimiter to split the text by.  The default is the " "
    :return:
    """

    freq_funct: Callable[[str], Dict[str, int]] = partial(count_freq_across_text, delimiter=delimiter)

    return reduce(combine_count_dict, map(freq_funct, docs))
