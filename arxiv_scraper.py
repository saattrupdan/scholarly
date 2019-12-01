def fetch(category = '', max_results = 5, start = 0):
    '''
    Fetch papers from the arXiv.

    INPUT
        category: str = ''
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
    import json

    if category:
        search_query = 'cat:' + category
    else:
        search_query = 'cat'

    params = {
        'search_query': search_query,
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
        id = entry.id.string
        authors = ','.join(name.string for author in entry.find_all('author')
                                       for name in author.find_all('name'))
        updated = entry.updated.string
        published = entry.published.string
        title = entry.title.string
        abstract = entry.summary.string
        cats = ','.join(cat['term'] for cat in entry.find_all('category'))

        papers.append({
        'id': id,
        'authors': authors,
        'updated': updated,
        'published': published,
        'title': title,
        'abstract': abstract,
        'categories': cats
        })

    return papers

def scrape(fname = 'arxiv.json', start = 0, batch_size = 1000):
    from itertools import count
    import json
    from time import sleep
    from tqdm import tqdm

    file_out = open(fname, 'a')
    pbar = tqdm(desc = 'Scraping papers')

    for idx in count(start = start, step = batch_size):
        batch = fetch(start = idx, max_results = batch_size)
        json.dump(batch, file_out)
        pbar.update(batch_size)
        sleep(5)

if __name__ == '__main__':
    scrape('/home/dn16382/pCloudDrive/public_folder/scholarly_data/arxiv.json')
