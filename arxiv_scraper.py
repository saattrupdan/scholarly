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
        self.engine = create_engine(f'sqlite:///{Path(data_dir) / db_name}')
        self.create_tables()
        self.populate_cats()

    def create_tables(self):
        ''' Create main table of the database. '''
        from sqlalchemy import MetaData, Table, Column
        from sqlalchemy import String, DateTime, ForeignKey

        metadata = MetaData()

        Table('papers', metadata,
            Column('id', String, primary_key = True),
            Column('updated', DateTime),
            Column('published', DateTime),
            Column('title', String),
            Column('abstract', String)
        )

        Table('master_cats', metadata,
            Column('id', String, primary_key = True),
            Column('name', String)
        )

        Table('cats', metadata,
            Column('id', String, primary_key = True),
            Column('name', String)
        )

        Table('cats_master_cats', metadata,
            Column('category_id', String, ForeignKey('cats.id'), 
                primary_key = True),
            Column('master_category_id', String,
                ForeignKey('master_cats.id'), primary_key = True)
        )

        Table('papers_cats', metadata,
            Column('paper_id', String, ForeignKey('papers.id'),
                primary_key = True),
            Column('category_id', String, ForeignKey('cats.id'),
                primary_key = True)
        )

        Table('authors', metadata,
            Column('id', String, primary_key = True)
        )

        Table('papers_authors', metadata,
            Column('paper_id', String, ForeignKey('papers.id'),
                primary_key = True),
            Column('author_id', String, ForeignKey('authors.id'),
                primary_key = True)
        )

        metadata.create_all(self.engine)
        return self

    def populate_cats(self):
        ''' Fetch list of all ArXiv categories from arxitics.com and
            use it to populate the cats, master_cats and cats_master_cats
            tables in the database. '''
        import requests
        from bs4 import BeautifulSoup
        from sqlalchemy.exc import IntegrityError

        master_cats = {
            'physics': 'Physics',
            'math': 'Mathematics',
            'cs': 'Computer Science',
            'q-bio': 'Quantitative Biology',
            'q-fin': 'Quantitative Finance',
            'stats': 'Statistics'
        }

        with self.engine.connect() as conn:
            for id, name in master_cats.items():
                try:
                    conn.execute(f'''
                        insert into master_cats
                        values ("{id}", "{name}");
                    ''')
                except IntegrityError:
                    pass

            base_url = 'http://arxitics.com/help/categories'
            for master_cat in master_cats:
                response = requests.get(base_url, {'group': master_cat})
                soup = BeautifulSoup(response._content, 'lxml')
                for li in soup.find_all('li'):
                    if li.strong is not None:
                        cat_id = li.strong.text
                        cat_name = li.span.text[2:]
                        try:
                            conn.execute(f'''
                                insert into cats
                                values ("{cat_id}", "{cat_name}");
                            ''')
                            conn.execute(f'''
                                insert into cats_master_cats
                                values ("{cat_id}", "{master_cat}");
                            ''')
                        except IntegrityError:
                            pass
        return self

    def insert_paper(self, paper_id: str, authors: str, updated, published, 
        title: str, abstract: str, categories: str, conn = None):
        ''' Insert a row into the database.

        INPUT
            paper_id: str
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

        queries = []
        queries.append(f'''
            insert into papers
            values ("{paper_id}", "{updated}", "{published}",
            "{title}", "{abstract}");
        ''')

        for author in authors.split(','):
            queries.append(f'''
                insert into authors 
                values ("{author.strip()}");
            ''')
            queries.append(f'''
                insert into papers_authors
                values ("{paper_id}", "{author}");
            ''')

        for category in categories.split(','):
            queries.append(f'''
                insert into papers_cats
                values ("{paper_id}", "{category.strip()}");
            ''')

        query = ' '.join(queries)

        if conn is None:
            with self.engine.connect() as conn:
                for query in queries:
                    try:
                        conn.execute(query)
                    except IntegrityError:
                        pass
        else:
            for query in queries:
                try:
                    conn.execute(query)
                except IntegrityError:
                    pass
        return self

    def table2dataframe(self, table_name):
        ''' Convert table to a Pandas DataFrame object. '''
        import pandas as pd
        with self.engine.connect() as conn:
            return pd.read_sql_table(table_name, conn)

def clean(doc: str):
    ''' Clean a document. '''
    import re

    # Remove newline symbols
    doc = re.sub('\n', ' ', doc)

    # Convert LaTeX equations of the form $...$, $$...$$, \[...\] or 
    # \(...\) to -EQN-
    dollareqn = '(?<!\$)\${1,2}(?!\$).*?(?<!\$)\${1,2}(?!\$)'
    bracketeqn = '\\[\[\(].*?\\[\]\)]'
    eqn = f'( {dollareqn} | {bracketeqn} )'
    doc = re.sub(eqn, '-EQN-', doc)

    # Remove scare quotes, both as " and \\"
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
        'paper_id': entry.id.string,
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
    from time import sleep
    from tqdm import tqdm
    from pathlib import Path

    # Create data directory
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        data_dir.mkdir()

    # Remove existing database if we are overwriting
    if overwrite:
        (data_dir / db_name).unlink()

    # Load database or create new one if it does not exist
    db = ArXivDatabase(db_name = db_name, data_dir = data_dir)

    # Get list of categories, sorted alphabetically
    with db.engine.connect() as conn:
        result = conn.execute('select cats.id from cats')
        cats = [cat[0] for cat in result]

    # Start from a given category
    if start_from is not None:
        try:
            cats = cats[cats.index(start_from):]
        except ValueError:
            pass

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
                        db.insert_paper(**row, conn = conn)

                cat_idx += len(batch)
                pbar.update(len(batch))
                sleep(5)

                # Add a strike if there was no results, or reset the
                # strikes if there was a result
                if len(batch):
                    strikes = 0
                else:
                    strikes += 1

if __name__ == '__main__':
    scrape()
