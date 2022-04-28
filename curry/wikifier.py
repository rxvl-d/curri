import os
import pickle
import socket
from functools import lru_cache
from urllib import parse, request, error
import json
import backoff

class Wikifier:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.user_key = os.getenv("WIKIFIER_KEY")
        self.wikifier_root = "http://www.wikifier.org/"
        self.max_text_size = 10000

    @lru_cache(maxsize=600)
    @backoff.on_exception(backoff.expo, (ConnectionResetError, socket.timeout, error.HTTPError))
    def wikify(self, text):
        lang = "de"
        threshold = 0.0
        data = parse.urlencode([
            ("text", text),
            ("lang", lang),
            ("userKey", self.user_key),
            ("wikiDataClassIds", "true"),
            ("support", "true"),
            ("ranges", "false"),
            ("includeCosines", "true"),
            ("pageRankSqThreshold", "%g" % threshold),
            ("wikiDataClasses", "true")
        ])
        url = self.wikifier_root + "annotate-article"
        req = request.Request(url, data=data.encode("utf8"), method="POST")
        with request.urlopen(req, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))
        return response

    def neighbours(self, title, nPredLevels, nSuccLevels):
        # Prepare the URL.
        data = parse.urlencode([
            ("lang", "en"),
            ("title", title),
            ("userKey", self.user_key),
            ("nPredLevels", nPredLevels),
            ("nSuccLevels", nSuccLevels)])
        url = self.wikifier_root + "get-neigh-graph?" + data
        # Call the Wikifier and read the response.
        with request.urlopen(url, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))

        return response

    def wikify_threshold(self, text, threshold):
        return {'annotations': list(filter(lambda a: a['pageRank'] > threshold, self.wikify(text)['annotations']))}

    def wikify_cached(self, urls):
        wikify_cache = self.get_cache()
        return [wikify_cache[url] for url in urls]

    def get_cache(self):
        with open(self.cache_dir + '/wikified.cache', 'rb') as f:
            return pickle.load(f)
