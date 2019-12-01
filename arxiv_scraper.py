def fetch(category, max_results = 5, start = 0):
    '''
    Fetch papers from the arXiv.

    INPUT
        category: str
            The name of the ArXiv category. Leave blank to search among all
            categories
        max_results: int = 5
            Maximal number of papers scraped, ArXiv limits this to 30,000
        start: int = 0
            The index of the paper from which the scraping begins

    OUTPUT
        A JSON string in which each entry has the following attributes:
            id
            authors
            updated
            published
            title
            abstract
            fields
    '''
    import requests
    from bs4 import BeautifulSoup

    params = {
        'search_query': 'cat:' + category,
        'start': start,
        'max_results': max_results,
        'sortBy': 'lastUpdatedDate',
        'sortOrder': 'descending'
    }

    api_url = 'http://export.arxiv.org/api/query'
    no_response = True
    while no_response:
        response = requests.get(api_url, params = params)
        soup = BeautifulSoup(response._content, 'lxml')
        no_response = soup.find_all('entry') == []

    papers = {
        'id': [],
        'authors': [],
        'updated': [],
        'published': [],
        'title': [],
        'abstract': [],
        'categories': []
    }

    papers = []
    for entry in soup.find_all('entry'):
        try:
            cats = ','.join(cat['term'] for cat in entry.find_all('category'))
            authors = ','.join(name.string 
                for author in entry.find_all('author')
                for name in author.find_all('name'))

            papers.append({
            'id': entry.id.string,
            'authors': authors,
            'updated': entry.updated.string,
            'published': entry.published.string,
            'title': entry.title.string,
            'abstract': entry.summary.string,
            'categories': cats
            })

        except:
            pass

    return papers

def scrape(fname = 'arxiv', data_dir = 'data', start = 0, 
    batch_size = 1000):
    from itertools import count, chain
    import pandas as pd
    import json
    from time import sleep
    from tqdm import tqdm
    import os

    # Create data directory
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # Get list of categories
    cat_path = os.path.join(data_dir, 'cats.tsv')
    if os.path.isfile(cat_path):
        cats_df = pd.read_csv(cat_path, sep = '\t')
    else:
        cats_df = get_cats(os.path.join(data_dir, 'cats'))
    cats = cats_df['cat']

    # Create empty JSON file
    path = os.path.join(data_dir, fname + '.json')
    with open(path, 'w') as f:
        json.dump([], f)

    # Scraping loop
    with tqdm(desc = 'Scraping papers') as pbar:
        for idx in count(start = start, step = batch_size):
            for cat in cats:

                # Get current batch
                batch = fetch(cat, start = idx, max_results = batch_size)

                # Load previous JSON data
                with open(path, 'r') as f:
                    json_data = json.load(f)

                # Concatenate the previous data with current batch
                json_data = list(chain(json_data, batch))

                # Save current JSON data
                with open(path, 'w') as f:
                    json.dump(json_data, f)

                # Give the API a break before continuing
                pbar.update(len(batch))
                sleep(5)

def get_cats(save_to = None):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    master_cats = [
        'physics',
        'math',
        'cs',
        'q-bio',
        'q-fin',
        'stats'
    ]

    data = {'cat': [], 'master_cat': [], 'desc': []}
    base_url = 'http://arxitics.com/help/categories'
    for master_cat in master_cats:
        response = requests.get(base_url, {'group': master_cat})
        soup = BeautifulSoup(response._content, 'lxml')
        for li in soup.find_all('li'):
            if li.strong is not None:
                data['cat'].append(li.strong.text)
                data['master_cat'].append(master_cat)
                if li.span is not None:
                    data['desc'].append(li.span.text[2:])
                else:
                    data['desc'].append('')

    df = pd.DataFrame(data)
    if save_to is not None:
        df.to_csv(save_to + '.tsv', sep = '\t', index = False)

    return df

if __name__ == '__main__':
    pcloud_dir = '/home/dn16382/pCloudDrive/public_folder/scholarly_data'
    scrape(data_dir = pcloud_dir, start = 0, batch_size = 10000)
