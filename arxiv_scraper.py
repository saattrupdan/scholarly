import datetime

class ArXivDatabase:
    ''' A SQLite databse for storing abstracts of ArXiv papers. 
    
    INPUT
        db_name: str = 'arxiv_data'
            Name of the database
        data_dir: str = 'data'
            Folder which contains the database
    '''

    def __init__(self, db_name: str = 'arxiv_data', data_dir: str = 'data'):
        from sqlalchemy import create_engine
        import os
        self.db_name = db_name
        path = os.path.join(data_dir, db_name)
        self.engine = create_engine(f'sqlite:///{path}')
        self.create_table()

    def create_table(self):
        ''' Create main table of the database. '''
        from sqlalchemy import MetaData, Table, Column
        from sqlalchemy import String, DateTime
        metadata = MetaData()
        Table(self.db_name, metadata,
            Column('id', String, primary_key = True),
            Column('authors', String),
            Column('updated', DateTime),
            Column('published', DateTime),
            Column('title', String),
            Column('abstract', String),
            Column('categories', String)
        )
        metadata.create_all(self.engine)
        return self

    def run_query(self, query: str):
        ''' Run any SQL query for the databse. '''
        with self.engine.connect() as conn:
            conn.execute(query)
        return self

    def insert_row(self, id: str, authors: str, updated: datetime.datetime, 
        published: datetime.datetime, title: str, abstract: str, 
        categories: str):
        ''' Insert a row into the database.

        INPUT
            id: str
                The unique ArXiv id. If a paper already exists with that
                id then the row will not be inserted
            authors: str
                Authors of the paper, separated by commas
            updated: datetime.datetime
                Date and time for when the paper was last updated
            published: datetime.datetime
                Date and time for when the paper was published onto the ArXiv
            title: str
                Title of the paper
            abstract: str
                Abstract of the paper
            categories: str
                The ArXiv categories that the paper is categorised as, 
                separated by commas
        '''
        from sqlalchemy.exc import IntegrityError
        query = f'''insert into {self.db_name}
                    values ("{id}", "{authors}", "{updated}", 
                            "{published}", "{title}", "{abstract}", 
                            "{categories}");'''
        try:
            self.run_query(query)
        except IntegrityError:
            pass

    def delete_row(self, id: str):
        ''' Delete row with a given ArXiv ID.

        INPUT
            id: str
                The ArXiv ID of the row to be deleted.
        '''
        query = f'delete from {self.db_name} where id = "{id}"'
        self.run_query(query)

    def update_row(self, id: str, **kwargs):
        ''' Update a row in the database with a given ArXiv ID.

        INPUT
            id: str
                The unique ArXiv id of the row to be updated.
            **kwargs
                Values to update
        '''
        changes = ', '.join(f'{col} = "{val}"' for col, val in kwargs.items())
        query = f''' update {self.db_name}
                     set {changes}
                     where id = "{id}";'''
        self.run_query(query)

    def print_all_data(self):
        ''' Print all the data in the database. '''
        with self.engine.connect() as conn:
            for row in conn.execute(f'select * from {self.db_name}'):
                print(row)
        return self

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
    from datetime import datetime

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
            'updated': datetime.fromisoformat(entry.updated.string),
            'published': datetime.fromisoformat(entry.published.string),
            'title': entry.title.string,
            'abstract': entry.summary.string,
            'categories': cats
            })

        except:
            pass

    return papers

def scrape(db_name = 'arxiv_data', data_dir = 'data', batch_size = 1000,
    patience = 10, max_papers_per_cat = None, overwrite = True,
    start_from = None):
    ''' Scrape papers from the ArXiv.

    INPUT
        db_name: str = 'arxiv_data'
            Name of the SQLite databse where the data will be stored
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

    db = ArXivDatabase(db_name = db_name, data_dir = data_dir)

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

    # Scraping loop
    with tqdm() as pbar:
        for cat in cats:
            pbar.set_description(f'Scraping {cat}')
            cat_idx, strikes = 0, 0
            while strikes <= patience:

                # Fetch data and store into databse
                batch = fetch(cat, start = cat_idx, max_results = batch_size)
                for row in batch:
                    db.insert_row(**row)

                cat_idx += len(batch)
                pbar.update(len(batch))
                sleep(5)

                # Strike management
                if len(batch):
                    strikes = 0
                else:
                    strikes += 1

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
    scrape()
