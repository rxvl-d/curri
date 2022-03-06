from functools import lru_cache

import yake
from tqdm import tqdm

from curry.clean import Cleaner

from sklearn.feature_extraction.text import CountVectorizer


class Extractor:
    def __init__(self, cleaner='tf'):
        self.cleaner = Cleaner(cleaner)

    def cleaned_content(self, contents):
        return [self.cleaner.clean(c) for c in tqdm(contents, desc='cleaning')]

    @lru_cache(maxsize=1000)
    def to_kws(self, content):
        ranked_kws = yake.KeywordExtractor(lan='de').extract_keywords(content)
        sorted_kws = sorted(ranked_kws, key=lambda x: x[1], reverse=True)
        return [kw for (kw, _) in sorted_kws]

    def keywords(self, contents):
        vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda x: x)
        kws = [self.to_kws(c) for c in tqdm(self.cleaned_content(contents), desc='keyword ex.')]
        return vectorizer.fit_transform(kws)
