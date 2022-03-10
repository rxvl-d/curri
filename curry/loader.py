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

    def sublessons_w_content(self, filter_multi_grade, land):
        df = self.sublessons()
        if filter_multi_grade:
            klass_range_agg = df.groupby(['land', 'url']).klass.apply(lambda s: s.max() - s.min())
            lessons_in_specific_grades = klass_range_agg[klass_range_agg <= 2].index
            filtered = df[df[['land', 'url']].apply(tuple, axis=1).isin(lessons_in_specific_grades)]
            # NOTE: since just returning the filtered DF right now would not result in the desired effect
            # of filtering features (due to caching), I"m returning ilcos and actual feature filtering
            # can happen later. Should be removable once the caches are cleared.
            selected_ilocs = self.get_ilocs_from_index(df.index, filtered.index)
        else:
            selected_ilocs = None

        if land:
            selected_land_ilocs = self.get_ilocs_from_index(df.index, df[df.land == land].index)
        else:
            selected_land_ilocs = None

        df['content'] = df.grundwissen_url.apply(self.read_file)
        return df, selected_ilocs, selected_land_ilocs

    def get_ilocs_from_index(self, source_index, filtered_index):
        selected_ilocs = []
        for idx in filtered_index:
            iloc = source_index.get_loc(idx)
            selected_ilocs.append(iloc)
        return selected_ilocs