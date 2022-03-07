import yake
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from curry.clean import Cleaner
from curry.utils import cache_file


class Extractor:
    def __init__(self, cleaner='tf'):
        self.cleaner = Cleaner(cleaner)

    @cache_file('.cleaned.cache')
    def cleaned_content(self, contents):
        return [self.cleaner.clean(c) for c in tqdm(contents, desc='cleaning')]

    def to_kws(self, content):
        ranked_kws = yake.KeywordExtractor(lan='de').extract_keywords(content)
        sorted_kws = sorted(ranked_kws, key=lambda x: x[1], reverse=True)
        return [kw for (kw, _) in sorted_kws]

    @cache_file('.keywords.cache')
    def keywords(self, contents):
        vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda x: x)
        kws = [self.to_kws(c) for c in tqdm(self.cleaned_content(contents), desc='keyword ex.')]
        return vectorizer.fit_transform(kws)

    def land_one_hot(self, lands):
        one_hot = OneHotEncoder()
        return one_hot.fit_transform([[l] for l in lands])

    def join(self, contents, lands):
        return hstack([self.keywords(contents), self.land_one_hot(lands)]).todense()

class SentenceTransformer:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def run(self, contents):
        all_sentences = []
        sentence_mapping = []
        for i, content in enumerate(contents):
            start_index = len(all_sentences)
            all_sentences += [s for s in content.split('.') if s.strip()]
            end_index = len(all_sentences)
            sentence_mapping.append((i, (start_index, end_index)))
        return self.embedder.encode(all_sentences), sentence_mapping
