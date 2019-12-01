def fetch(category, max_results = 5, start = 0):
    ''' Fetch papers from the arXiv

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
            categories
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
    response = requests.get(api_url, params = params)
    soup = BeautifulSoup(response._content, 'lxml')

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

def scrape(fname = 'arxiv', data_dir = 'data', batch_size = 1000,
    patience = 10, max_papers_per_cat = None, overwrite = True,
    start_from = None):
    ''' Scrape papers from the ArXiv.

    INPUT
        fname: str = 'arxiv'
            Name of the JSON file where the data will be stored, without
            file extension
        data_dir: str = 'data'
            Directory in which the data files are to be stored
        batch_size: int = 0
            The amount of papers fetched at each GET request. Must be at
            most 10,000
        patience: int = 10
            The amount of successive failed GET requests before moving on
            to the next category
        max_papers_per_cat: int or None = None
            The maximum number of papers to fetch for each category
        overwrite: bool = True
            Whether the JSON file should be overwritten
        start_from: str or None = None
            A category to start from, where None means start from scratch
    '''
    from itertools import count, chain
    import pandas as pd
    import json
    from time import sleep
    from tqdm import tqdm
    import os

    # Create data directory
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # Get list of categories, sorted alphabetically
    cat_path = os.path.join(data_dir, 'cats.tsv')
    if os.path.isfile(cat_path):
        cats_df = pd.read_csv(cat_path, sep = '\t')
    else:
        cats_df = get_cats(os.path.join(data_dir, 'cats'))
    cats = sorted(cats_df['cat'])

    # Start from a given category
    if start_from is not None:
        try:
            cats = cats[cats.index(start_from):]
        except ValueError:
            pass

    # Create empty JSON file
    json_path = os.path.join(data_dir, fname + '.json')
    if overwrite:
        with open(json_path, 'w') as f:
            json.dump([], f)

    # Scraping loop
    with tqdm() as pbar:
        for cat in cats:
            pbar.set_description(f'Scraping {cat}')
            cat_idx, strikes = 0, 0
            while strikes <= patience:
                batch = fetch(cat, start = cat_idx, max_results = batch_size)
                if len(batch):
                    # Reset strikes
                    strikes = 0

                    # Load previous data from JSON file
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)

                    # Concatenate the new data and save to new JSON file
                    json_data = list(chain(json_data, batch))
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f)
                else:
                    strikes += 1

                cat_idx += len(batch)
                pbar.update(len(batch))
                sleep(5)

def get_cats(save_to = None):
    ''' Fetch list of all ArXiv categories from arxitics.com
    
    INPUT
        save_to: str or None
            File name to save the dataframe to, without file extension

    OUTPUT
        A Pandas DataFrame object with the categories
    '''
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
    scrape(data_dir = 'data', batch_size = 1000, start_from = None)
