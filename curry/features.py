import json
import logging
import pickle

import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from curry.babelfy import Babelfier
from curry.clean import Cleaner
from curry.wikifier import Wikifier
from curry.yakekw import Yaker


class Extractor:
    def __init__(self, cache_dir):
        self.cleaner = Cleaner(cache_dir)
        self.wikifier = Wikifier(cache_dir)
        self.babelfier = Babelfier(cache_dir)
        self.sentence_transformer = SentenceTransformer(cache_dir)
        self.yake = Yaker(cache_dir)

    def cleaned_content(self, urls):
        return self.cleaner.clean_cached(urls)

    def keywords(self, urls):
        return self.count_vectorize(self.yake.kw_cache(urls))

    def sentence_transformers(self, urls):
        return self.sentence_transformer.cached_vecs(urls)

    def tfidf(self, urls):
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('german'))
        out = vectorizer.fit_transform(self.cleaned_content(urls))
        return out, vectorizer.get_feature_names_out()

    def babelfy_kws(self, urls):
        threshold = 0.5
        vecs, bnids = self.count_vectorize(
            [[a['babelSynsetID'] for a in annotations if a['score'] > threshold]
             for annotations in self.babelfier.bab_cached(urls)])
        bnid_to_description_map = self.babelfier.bnid_to_description_map()
        return vecs, [json.dumps(bnid_to_description_map[bnid], cls=SetEncoder) for bnid in bnids]

    def wikifier_kws(self, urls):
        threshold = 0.01
        return self.count_vectorize(
            [[a['title'] for a in wikified['annotations'] if a['pageRank'] > threshold]
             for wikified in self.wikifier.wikify_cached(urls)])

    def land_one_hot(self, lands):
        one_hot = OneHotEncoder()
        return one_hot.fit_transform([[l] for l in lands])

    def content_vecs(self, urls, vec_type):
        features = None
        if vec_type == 'kw':
            content_vec, features = self.keywords(urls)
        elif vec_type == 'babelkw':
            content_vec, features = self.babelfy_kws(urls)
        elif vec_type == 'wikikw':
            content_vec, features = self.wikifier_kws(urls)
        elif vec_type == 'st':
            content_vec = self.sentence_transformers(urls)
        elif vec_type == 'tfidf':
            content_vec, features = self.tfidf(urls)
        else:
            raise Exception(f"Unknown vector type: {vec_type}")
        return content_vec, features

    def concatenate_hetero_arrays(self, arrs):
        types = set(map(type, arrs))
        if types == {csr_matrix}:
            return hstack(arrs).tocsr()
        elif types == {csr_matrix, np.ndarray}:
            to_concatenate = [(arr if type(arr) == csr_matrix else csr_matrix(arr)) for arr in arrs]
            return hstack(to_concatenate).tocsr()
        else:
            raise Exception(f"Unexpected combination of array types {types}")

    def count_vectorize(self, keywords):
        vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda x: x)
        return vectorizer.fit_transform(keywords), vectorizer.get_feature_names_out()


class SentenceTransformer:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def run(self):
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        all_sentences = []
        sentence_mapping = []
        with open(self.cache_dir + 'cleaned_content.cache', 'rb') as f:
            contents = pickle.load(f)
            for i, url in enumerate(contents):
                content = contents[url]
                start_index = len(all_sentences)
                all_sentences += [s for s in content.split('.') if s.strip()]
                end_index = len(all_sentences)
                sentence_mapping.append((url, (start_index, end_index)))
        logging.info('Starting Encoding')
        encoded = embedder.encode(all_sentences)
        logging.info('Finished Encoding')
        return self.doc_vecs(encoded, sentence_mapping)

    def doc_vecs(self, sentence_vecs, mapping):
        vecs = dict()
        for (url, (start, end)) in mapping:
            vecs[url] = np.mean(sentence_vecs[start:end, :], axis=0)
        return vecs

    def write_cache(self):
        with open(self.cache_dir + 'sentence_transformers.cache', 'wb') as f:
            pickle.dump(self.run(), f)

    def cached_vecs(self, urls):
        with open(self.cache_dir + 'cleaned_content.cache', 'rb') as f:
            cache = pickle.load(f)
            return [cache[url] for url in urls]


class SetEncoder(json.JSONEncoder):
   def default(self, obj):
      if isinstance(obj, set):
         return list(obj)
      return json.JSONEncoder.default(self, obj)
