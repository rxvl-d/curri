from functools import lru_cache

import trafilatura
import json
import readabilipy

class Cleaner:
    def __init__(self, cleaner_type):
        assert cleaner_type in {'tf', 'rd'}
        self.cleaner_type = cleaner_type

    def trafilatura_lines(self, html_content):
        tf = trafilatura.extract(html_content,
                                   output_format='json',
                                   include_comments=False,
                                   include_images=True,
                                   include_links=True,
                                   include_tables=True,
                                   with_metadata=True)
        return json.loads(tf)['raw_text'].split('.')

    def readability_lines(self, html_content):
        return [line['text'] for line in readabilipy.simple_json_from_html_string(html_content)['plain_text']]

    def remove_equations(self, extracted_lines):
        accepted_lines = []
        for line in extracted_lines:
            if '\\' in line:
                pass
            else:
                accepted_lines.append(line)
        return '. '.join(accepted_lines)

    def clean(self, html_content):
        if self.cleaner_type == 'tf':
            return self.remove_equations(self.trafilatura_lines(html_content))
        elif self.cleaner_type == 'rd':
            return self.remove_equations(self.readability_lines(html_content))
        else:
            raise Exception('boom')