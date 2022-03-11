import json

import readabilipy
import trafilatura


class Cleaner:
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
        main_content = trafilatura.extract(html_content, favor_recall=True)
        main_lines = []
        for line in self.readability_lines(html_content):
            if line in main_content:
                main_lines.append(line)
        return self.remove_equations(main_lines)