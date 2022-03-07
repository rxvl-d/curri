import os
import socket
from functools import lru_cache
from urllib import parse, request, error
import json
import backoff

class Wikifier:
    def __init__(self):
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

def test():
    w = Wikifier()
    return w.wikify("A right isoceles triangle has a hypotenuse of 20 feet. What are the lengths of the legs of the triangle?")