from functools import lru_cache

import pandas as pd

class Loader:
    def __init__(self, data_dir='../data/'):
        self.data_dir = data_dir

    @lru_cache(maxsize=1)
    def lessons(self):
        return pd.read_csv(self.data_dir + 'lessons.csv', index_col=0)

    @lru_cache(maxsize=1)
    def detail_pages(self):
        return pd.read_csv(self.data_dir + 'detail_pages.csv', index_col=0)

    @lru_cache(maxsize=1)
    def sublessons(self):
        out= pd.merge(
            self.lessons(),
            self.detail_pages(),
            left_on='url',
            right_on='lesson_url'
        ).drop(
            columns='lesson_url'
        )
        return out[out.grundwissen_url != '/waermelehre/wetter-und-klima/grundwissen/strahlungshaushalt-der-erde'] # Problematic lesson to parse

    @lru_cache(maxsize=560)
    def read_file(self, url):
        with open(self.data_dir + 'grundwissen_pages/' + url.replace('/', 'SLASH')) as f:
            return f.read()

    @lru_cache(maxsize=1)
    def sublessons_w_content(self):
        df = self.sublessons()
        df['content'] = df.grundwissen_url.apply(self.read_file)
        return df