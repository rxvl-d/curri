import pickle
from functools import lru_cache

import sparql_dataframe
from tqdm import tqdm

from curry.babelfy import Babelfier
from curry.wikifier import Wikifier


class DBPediaPhysicsResources:
    def __init__(self, data_dir='../data/'):
        self.data_dir = data_dir
        self.cache_dir = data_dir + 'cache/'
        self.babelfier = Babelfier(self.cache_dir)
        self.wikifier = Wikifier(self.cache_dir)

    def categories_for(self, dbpedia_resource_url):
        return self.categories_by_dbpedia_iri_cached().get(dbpedia_resource_url)


    @lru_cache(maxsize=20000)
    def get_rdf_type(self, resource, depth):
        endpoint = "http://dbpedia.org/sparql"

        q = """
            prefix dbr: <http://dbpedia.org/resource/> 
            prefix dbo: <http://dbpedia.org/ontology/>
            prefix skos: <http://www.w3.org/2004/02/skos/core#>

            select distinct ?subcategory  where {
              <""" + resource + """> dbo:wikiPageRedirects* ?redirect.
              ?redirect <http://purl.org/dc/terms/subject> ?category.
              ?category """ + '?/'.join(['skos:broader' for _ in range(depth)]) + """ ?subcategory
            }
        """
        df = sparql_dataframe.get(endpoint, q)
        return df.subcategory.values

    def get_categories_uncached(self, dbpedia_resource_urls, depth):
        categories = [self.get_rdf_type(r, depth) for r in tqdm(dbpedia_resource_urls)]
        return categories

    @lru_cache(maxsize=2)
    def relevant_annotations(self, annotation_source='babel'):
        categories_by_iri = self.categories_by_dbpedia_iri_cached()
        if annotation_source == 'babel':
            return self.get_physics_relevant_babelfy_annotations(categories_by_iri)
        elif annotation_source == 'wikifier':
            return self.get_physics_relevant_wikifier_annotations(categories_by_iri)
        else:
            raise Exception("boom")

    def get_physics_relevant_babelfy_annotations(self, categories_by_iri):
        filtered = dict()
        babelfy_cache = self.babelfier.get_cache()
        for url in babelfy_cache:
            out = []
            for baby_ann in babelfy_cache[url]:
                if baby_ann.get('DBpediaURL'):  # and baby_ann['score'] > 0.5:
                    if any([self.is_accepted_topics(c)
                            for c in categories_by_iri[baby_ann['DBpediaURL']]]):
                        out.append(baby_ann['DBpediaURL'])
            filtered[url] = out
        return filtered

    def get_physics_relevant_wikifier_annotations(self, categories_by_iri):
        filtered = dict()
        wikifier_cache = self.wikifier.get_cache()
        for url in wikifier_cache:
            out = []
            for wiki_ann in wikifier_cache[url]['annotations']:
                if wiki_ann.get('dbPediaIri') and wiki_ann['cosine'] > 0.2:  # The cosine filter is simply required to make sure we aren't grabbing too much
                    if any([self.is_accepted_topics(c) for c in categories_by_iri[wiki_ann['dbPediaIri']]]):
                        out.append(wiki_ann['dbPediaIri'])
            filtered[url] = out
        return filtered

    def is_accepted_topics(self, c):
        return ('http://dbpedia.org/resource/Category:Subfields_of_physics' in c) or \
               ('http://dbpedia.org/resource/Category:Physics' in c) or \
               ('http://dbpedia.org/resource/Category:Concepts_in_physics' in c) or \
               ('http://dbpedia.org/resource/Category:Physical_sciences' in c) or \
               ('http://dbpedia.org/resource/Category:Electromagnetism' in c) or \
               ('http://dbpedia.org/resource/Category:Electrical_engineering' in c) or \
               ('http://dbpedia.org/resource/Category:Physical_quantities' in c) or \
               ('http://dbpedia.org/resource/Category:Universe' in c) or \
               ('http://dbpedia.org/resource/Category:Metrology' in c) or \
               ('http://dbpedia.org/resource/Category:Classical_mechanics' in c) or \
               ('http://dbpedia.org/resource/Category:Engineering_disciplines' in c) or \
               ('http://dbpedia.org/resource/Category:Applied_and_interdisciplinary_physics' in c) or \
               ('http://dbpedia.org/resource/Category:Applied_sciences' in c) or \
               ('http://dbpedia.org/resource/Category:Engineering_disciplines' in c)

    @lru_cache(maxsize=1)
    def categories_by_dbpedia_iri_cached(self):
        with open(self.cache_dir + 'categories_depth_5.pkl', 'rb') as f:
            return pickle.load(f)