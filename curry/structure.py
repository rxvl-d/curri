from bs4 import BeautifulSoup

def insert_under(new_node, parent_node, tree):
    for k in tree:
        if k == parent_node:
            tree[k].append 

def get_heading_structure_and_contents_by_heading(content_thema):
    current_section_heading = None
    current_section_heading_level = None
    under_heading = dict()
    heading_tree = dict()
    for el in content_thema.find_all():
        if (el.name == 'p') and current_section_heading:
            under_heading[current_section_heading] = \
                under_heading.get(current_section_heading, []) + [el.get_text().strip()]
        elif (el.name.startswith('h')) and current_section_heading:
            current_level = int(el.name[1:])
            if current_level > current_section_heading_level:
                insert_under(el.get_text().strip(), current_section_heading)


def get_contents(contents):
    soup = BeautifulSoup(contents, features='lxml')
    page_content = soup.select_one('.page-content')
    teaser_container = page_content.select_one('.teaser_container')
    topic_banner = page_content.select_one('.topic_banner')
    topic_name = topic_banner.get_text().strip()
    content_thema = page_content.select_one('.content-thema')
    lesson_name_el = teaser_container.select_one('h2')
    content_h1s = content_thema.select('h1')
    content_h2s = content_thema.select('h2')
    content_h3s = content_thema.select('h3')
    content_h4s = content_thema.select('h4')
    content_h5s = content_thema.select('h5')
    content_h6s = content_thema.select('h6')
    lesson_name = lesson_name_el.get_text().strip()
    das_wichtigste_points = get_das_wichtigste_points(teaser_container)
    image_and_text, remaining_tables = get_tables(content_thema)
    images, figures = get_images_and_figures(content_thema)
    animations = get_animations(content_thema)
    return dict(
        topic_name=topic_name,
        lesson_name=lesson_name,
        content_thema=content_thema,
        das_wichtigste_points=das_wichtigste_points,
        image_and_text=image_and_text or None,
        remaining_tables=remaining_tables or None,
        images=images or None,
        figures=figures or None,
        tasks_box=get_tasks_box(page_content) or None,
        animations=animations or None,
        content_h1s=content_h1s or None,
        content_h2s=content_h2s or None,
        content_h3s=content_h3s or None,
        content_h4s=content_h4s or None,
        content_h5s=content_h5s or None,
        content_h6s=content_h6s or None,

    )


def get_tasks_box(page_content):
    return page_content.select('.tasks_box')

def get_tables(page_content):
    contents = []
    all_tables = page_content.select('table')
    remaining_tables = []
    for table in all_tables:
        rows = table.select('tr')
        if (len(rows) == 1) and (len(rows[0].select('td')) == 2):
            l, r = rows[0].select('td')
            text_content = l.find('p') or r.find('p')
            img_content = l.find('img') or r.find('img')
            if text_content and img_content:
                contents.append((img_content, text_content))
            else:
                remaining_tables.append(table)
        else:
            remaining_tables.append(table)

    return contents, remaining_tables


def get_images_and_figures(page_content):
    images = []
    figures = []
    for image in page_content.select('img'):
        maybe_figure = image.find_parent('figure')
        if maybe_figure:
            figures.append((maybe_figure.find('figcaption'), image))
        else:
            images.append(image)
    return images, figures

def get_das_wichtigste_points(teaser_container):
    das_wichtigste = teaser_container.select_one('ul')
    das_wichtigste_points = None
    if das_wichtigste:
        for point in das_wichtigste.select('li'):
            if das_wichtigste_points is None:
                das_wichtigste_points=[]
            das_wichtigste_points.append(point.get_text().strip())
    return das_wichtigste_points

def get_animations(page_content):
    return page_content.select('.animation')

def show_tables(tables):
    from IPython.core.display import display, HTML
    out_str = ""

    for i, tables in enumerate(tables):
        html_str = f'<table><tr><td>{i}</td><td>'
        html_str += '</td><td>'.join([str(t) for t in tables])
        html_str += '</td></tr></table>'
        out_str += html_str

    display(HTML(out_str))


NO_HEADING = 'NO_HEADING'


def get_contents_detailed(content_thema):
    last_heading = NO_HEADING
    paras_with_heading = dict(NO_HEADING=[])
    figures = dict(NO_HEADING=[])
    for i in content_thema.find_all():
        if i.name.startswith('h'):
            last_heading = i.get_text().strip()
            paras_with_heading[last_heading] = []
            figures[last_heading] = []
        if i.name == 'p':
            paras_with_heading[last_heading].append(i.get_text())
        if i.name == 'figure':
            image = i.select_one('img')
            img_src = image.attrs['src'] if image else None
            fig_caption_el = i.find('figcaption')
            fig_caption = fig_caption_el.get_text() if fig_caption_el else None
            figures[last_heading].append((img_src, fig_caption))
        if i.name == 'figcaption':
            fig_caption = i.get_text().strip()
            figures[last_heading].append((None, fig_caption))
    return dict(paras_with_heading=paras_with_heading, figures=figures)


def show_paras_with_heading(paras_with_heading):
    for heading in paras_with_heading:
        print('=' * len(heading))
        print(heading)
        print('=' * len(heading))
        for para in paras_with_heading[heading]:
            print(para)
            print('-' * 40)


def show_figures_with_heading(figures):
    for heading in figures:
        print('=' * len(heading))
        print(heading)
        print('=' * len(heading))
        for fig_src, fig_caption in figures[heading]:
            print(fig_src)
            print(fig_caption)
            print('-' * 40)

def show_detailed(contents):
    soup = BeautifulSoup(contents)
    contents = get_contents_detailed(soup.select_one('.content-thema'))
    paras = contents['paras_with_heading']
    figures = contents['figures']
    print("PARAS:")
    show_paras_with_heading(paras)
    print("FIGURES:")
    show_figures_with_heading(figures)


def get_structured_text_content(html_content):
    soup = BeautifulSoup(html_content, features='lxml')
    contents = get_contents_detailed(soup.select_one('.content-thema'))
    out = dict()
    for heading in contents['paras_with_heading']:
        out[heading] = dict()
        out[heading]['figure_captions'] = \
            [fig_caption for _, fig_caption in contents['figures'].get(heading, []) if fig_caption]
        out[heading]['paras'] = contents['paras_with_heading'][heading]
    return out
