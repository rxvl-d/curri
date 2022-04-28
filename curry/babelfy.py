import logging
import os
import pickle
import socket
from urllib import error
from urllib.error import HTTPError

import backoff
from babelpy.babelfy import BabelfyClient

logging.basicConfig(level=logging.INFO)


def fatal_code(e):
    return 400 <= e.status < 500


class Babelfier():
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.api_key = os.getenv("BABELFY_KEY")
        self.babel_client = BabelfyClient(self.api_key)
        self.cache_file = '.babelfier_bab.cache'
        if not os.path.isfile(self.cache_file):
            self.write_cache(dict())
        with open(self.cache_file, 'rb') as f:
            self.cache = pickle.load(f)

    def write_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache, f)

    @backoff.on_exception(backoff.expo,
                          (ConnectionResetError, socket.timeout, error.HTTPError),
                          max_tries=3,
                          giveup=fatal_code)
    def bab(self, text):
        if text in self.cache:
            return self.cache[text]
        else:
            params = dict()
            params['lang'] = 'DE'
            params['text'] = text
            try:
                self.babel_client.babelfy(text, params)
                out = self.babel_client.entities
            except HTTPError as e:
                if e.code == 414:
                    out = self.bab(text[:len(text) // 2]) + self.bab(text[len(text) // 2:])
                else:
                    logging.error("Unexpected Failure but continuing. Failed text here: /tmp/unexpected_failure")
                    with open('/tmp/unexpected_failure', 'w') as f:
                        f.write(text)
                    out = []
            self.cache[text] = out
            self.write_cache(self.cache)
            return out

    def bab_cached(self, urls):
        babelfy_cache = self.get_cache()
        return [babelfy_cache[url] for url in urls]

    def get_cache(self):
        with open(self.cache_dir + '/babelfied.cache', 'rb') as f:
            return pickle.load(f)

    def bnid_to_description_map(self):
        out = dict()
        with open(self.cache_dir + '/babelfied.cache', 'rb') as f:
            babelfy_cache = pickle.load(f)
            for annotations in babelfy_cache.values():
                for annotation in annotations:
                    dbpedia = annotation.get('DBpediaURL') or None
                    text = annotation['text']
                    bab_id = annotation['babelSynsetID']
                    if out.get(bab_id) is None:
                        out[bab_id] = {'dbpedia': set(), 'text': {text}}
                        if dbpedia:
                            out[bab_id]['dbpedia'].add(dbpedia)
                    else:
                        if dbpedia:
                            out[bab_id]['dbpedia'].add(dbpedia)
                        out[bab_id]['text'].add(text)
        return out