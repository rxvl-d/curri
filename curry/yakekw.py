import pickle
import yake
from tqdm import tqdm


class Yaker:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def cache(self):
        yake_kws = dict()
        y = yake.KeywordExtractor(lan='de')
        with open(self.cache_dir + '/cleaned_content.cache', 'rb') as f:
            cleaned_content_cache = pickle.load(f)
        for url in tqdm(cleaned_content_cache):
            ranked_kws = y.extract_keywords(cleaned_content_cache[url])
            sorted_kws = sorted(ranked_kws, key=lambda x: x[1], reverse=True)
            yake_kws[url] = [kw for (kw, _) in sorted_kws]
        with open(self.cache_dir + '/yake.cache', 'wb') as f:
            pickle.dump(yake_kws, f)

    def kw_cache(self, urls):
        with open(self.cache_dir + '/yake.cache', 'rb') as f:
            cache = pickle.load(f)
            return [cache[url] for url in urls]

if __name__ == '__main__':
    Yaker('../data/cache').cache()