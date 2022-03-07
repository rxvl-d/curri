import numpy as np
import yake
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from nltk.corpus import stopwords

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

    @cache_file('.doc_vecs.cache')
    def sentence_transformers(self, contents):
        st = SentenceTransformer()
        out = np.array(st.doc_vecs(st.run()))
        assert len(out) == len(contents)
        return out

    @cache_file('.tfidf.cache')
    def tfidf(self, contents):
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('german'))
        return vectorizer.fit_transform(contents)

    def land_one_hot(self, lands):
        one_hot = OneHotEncoder()
        return one_hot.fit_transform([[l] for l in lands])

    def join(self, contents, lands, vec_type):
        land_vec_sparse = self.land_one_hot(lands)
        if vec_type == 'kw':
            content_vec = self.keywords(contents)
            return hstack([content_vec, land_vec_sparse]).todense()
        elif vec_type == 'st':
            content_vec = self.sentence_transformers(contents)
            return np.concatenate([content_vec, land_vec_sparse.todense()], axis=1)
        elif vec_type == 'tfidf':
            content_vec = self.tfidf(contents)
            return hstack([content_vec, land_vec_sparse]).todense()
        else:
            raise Exception("Boom!")

class SentenceTransformer:
    @cache_file('.sentence_transformer.cache')
    def run(self, contents):
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        all_sentences = []
        sentence_mapping = []
        for i, content in enumerate(contents):
            start_index = len(all_sentences)
            all_sentences += [s for s in content.split('.') if s.strip()]
            end_index = len(all_sentences)
            sentence_mapping.append((i, (start_index, end_index)))
        return embedder.encode(all_sentences), sentence_mapping

    def doc_vecs(self, run_outs):
        sentence_vecs, mapping = run_outs
        vecs = []
        for (idx, (start, end)) in mapping:
            vecs.append(np.mean(sentence_vecs[start:end, :], axis=0))
        return vecs
