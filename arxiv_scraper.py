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
        from pathlib import Path
        self.db_name = db_name
        self.engine = create_engine(f'sqlite:///{Path(data_dir) / db_name}')
        self.create_table()

    def create_table(self):
        ''' Create main table of the database. '''
        from sqlalchemy import MetaData, Table, Column, String, DateTime
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

    def insert_row(self, id: str, authors: str, updated, published, 
        title: str, abstract: str, categories: str, conn = None):
        ''' Insert a row into the database.

        INPUT
            id: str
                The unique ArXiv id. If a paper already exists with that id
                then the row will not be inserted
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
            conn: sqlalchemy.Connection = None
                A connection to execute the query, which defaults to 
                opening a new connection
        '''
        from sqlalchemy.exc import IntegrityError
        query = (f'insert into {self.db_name} '
                 f'values ("{id}", "{authors}", "{updated}", '
                 f'"{published}", "{title}", "{abstract}", '
                 f'"{categories}")')
        try:
            if conn is None:
                with self.engine.connect() as conn:
                    conn.execute(query)
            else:
                conn.execute(query)
        except IntegrityError:
            pass
        return self

    def delete_row(self, id: str, conn = None):
        ''' Delete row with a given ArXiv ID.

        INPUT
            id: str
                The ArXiv ID of the row to be deleted.
            conn: sqlalchemy.Connection = None
                The connection executing the query, which defaults to 
                opening a new connection
        '''
        query = f'delete from {self.db_name} where id = "{id}"'
        if conn is None:
            with self.engine.connect() as conn:
                conn.execute(query)
        else:
            conn.execute(query)
        return self

    def update_row(self, id: str, conn = None, **kwargs):
        ''' Update a row in the database with a given ArXiv ID.

        INPUT
            id: str
                The unique ArXiv id of the row to be updated.
            conn = None
                A Connection object, which will execute the query, which
                defaults to opening a new connection
            **kwargs
                Values to update
        '''
        changes = ', '.join(f'{col} = "{val}"' for col, val in kwargs.items())
        query = (f'update {self.db_name}'
                 f'set {changes}'
                 f'where id = "{id}";')
        if conn is None:
            with self.engine.connect() as conn:
                conn.execute(query)
        else:
            conn.execute(query)
        return self

    def to_dataframe(self):
        ''' Convert database to a Pandas DataFrame object. '''
        import pandas as pd
        with self.engine.connect() as conn:
            return pd.read_sql_table(self.db_name, conn)

def clean(doc: str):
    ''' Clean a document. '''
    import re

    # Remove newline symbols
    doc = re.sub('\n', ' ', doc)

    # Convert equations like $...$, $$...$$, $\[...\]$ or $\(...\)$ to -EQN-
    dollareqn = '(?<!\$)\${1,2}(?!\$).*?(?<!\$)\${1,2}(?!\$)'
    bracketeqn = '\\[\[\(].*?\\[\]\)]'
    eqn = f'( {dollareqn} | {bracketeqn} )'
    doc = re.sub(eqn, '-EQN-', doc)

    # Remove scare quotes
    doc = re.sub('(\\"|")', '', doc)

    # Merge multiple spaces
    doc = re.sub(r' +', ' ', doc)

    return doc.strip()

def fetch(category: str, max_results: int = 5, start: int = 0):
    ''' Fetch papers from the arXiv.

    INPUT
        category: str
            The name of the ArXiv category. Leave blank to search among all
            categories
        max_results: int = 5
            Maximal number of papers scraped, ArXiv limits this to 30,000
        start: int = 0
            The index of the paper from which the scraping begins

    OUTPUT
        A list of dictionaries representing each paper entry, with each
        dictionary having the following attributes:
            id: str
                The unique ArXiv identifier
            authors: str
                The authors of the paper, separated by commas
            updated: datetime.datetime
                Last updated date and time
            published: datetime.datetime
                Date and time when the paper was published on ArXiv
            title: str
                Title of the paper
            abstract: str
                Abstract of the paper
            categories: str
                The ArXiv categories that the paper falls under, separated
                by commas
    '''
    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime
    from time import sleep

    params = {
        'search_query': 'cat:' + category,
        'start': start,
        'max_results': max_results,
        'sortBy': 'lastUpdatedDate',
        'sortOrder': 'descending'
    }

    # Perform GET request
    while True:
        try:
            api_url = 'http://export.arxiv.org/api/query'
            response = requests.get(api_url, params = params)
            soup = BeautifulSoup(response._content, 'lxml')
            break
        except requests.exceptions.ConnectionError:
            sleep(1)
            continue

    papers = {
        'id': [],
        'authors': [],
        'updated': [],
        'published': [],
        'title': [],
        'abstract': [],
        'categories': []
    }

    # Convert data formats and store it in a list
    papers = []
    for entry in soup.find_all('entry'):
        cats = ','.join(cat['term'] for cat in entry.find_all('category'))
        authors = ','.join(clean(name.string) 
            for author in entry.find_all('author')
            for name in author.find_all('name'))

        papers.append({
        'id': entry.id.string,
        'authors': authors,
        'updated': datetime.fromisoformat(entry.updated.string[:-1]),
        'published': datetime.fromisoformat(entry.published.string[:-1]),
        'title': clean(entry.title.string),
        'abstract': clean(entry.summary.string),
        'categories': cats
        })

    return papers

def scrape(db_name: str = 'arxiv_data', data_dir: str = 'data', 
    batch_size: int = 1000, patience: int = 20, overwrite: bool = False, 
    max_papers_per_cat: int = None, start_from: int = None):
    ''' Scrape papers from the ArXiv.

    INPUT
        db_name: str = 'arxiv_data'
            Name of the SQLite databse where the data will be stored
        data_dir: str = 'data'
            Directory in which the data files are to be stored
        batch_size: int = 0
            The amount of papers fetched at each GET request - has to be 
            below 10,000
        patience: int = 20
            The amount of successive failed GET requests before moving on
            to the next category. The ArXiv API usually times out, resulting
            in a failed GET request, so this number should be reasonably
            large to rule these timeouts out
        overwrite: bool = False
            Whether the database file should be overwritten
        max_papers_per_cat: int = None
            The maximum number of papers to fetch for each category
        start_from: str = None
            A category to start from, which defaults to starting from scratch
    '''
    import pandas as pd
    from time import sleep
    from tqdm import tqdm
    from pathlib import Path

    # Create data directory
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        data_dir.mkdir()

    # Get list of categories, sorted alphabetically
    cat_path = data_dir / 'cats.tsv'
    if cat_path.is_file():
        cats_df = pd.read_csv(cat_path, sep = '\t')
    else:
        cats_df = get_cats(cat_path)
    cats = sorted(cats_df['cat'])

    # Start from a given category
    if start_from is not None:
        try:
            cats = cats[cats.index(start_from):]
        except ValueError:
            pass

    # Remove existing database if we are overwriting
    if overwrite:
        (data_dir / db_name).unlink()

    # Load database
    db = ArXivDatabase(db_name = db_name, data_dir = data_dir)

    # Scraping loop
    for cat in tqdm(cats, desc = 'Scraping ArXiv categories'):
        with tqdm(leave = False) as pbar:
            pbar.set_description(f'Scraping {cat}')
            cat_idx, strikes = 0, 0
            while strikes <= patience:

                # Fetch data
                batch = fetch(
                    category = cat, 
                    max_results = batch_size,
                    start = cat_idx
                )

                # Store data in database
                with db.engine.connect() as conn:
                    for row in batch:
                        db.insert_row(**row, conn = conn)

                cat_idx += len(batch)
                pbar.update(len(batch))
                sleep(5)

                # Add a strike if there was no results, or reset the
                # strikes if there was a result
                if len(batch):
                    strikes = 0
                else:
                    strikes += 1

def get_cats(save_to: str = None):
    ''' Fetch list of all ArXiv categories from arxitics.com
    
    INPUT
        save_to: str
            File name to save the dataframe to, with .tsv file extension

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
        df.to_csv(save_to, sep = '\t', index = False)

    return df

if __name__ == '__main__':
    scrape(start_from = 'cs.CY')
