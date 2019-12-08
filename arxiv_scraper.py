class ArXivDatabase:
    ''' A SQLite databse for storing ArXiv papers. 
    
    INPUT
        name: str = 'arxiv_data.db'
            Name of the database
        data_dir: str = 'data'
            Folder which contains the database
    '''

    def __init__(self, name: str = 'arxiv_data.db', data_dir: str = 'data'):
        from sqlalchemy import create_engine
        from pathlib import Path
        self.engine = create_engine(f'sqlite:///{Path(data_dir) / name}')
        self.create_tables()
        self.populate_cats()

    def create_tables(self):
        ''' Create the tables of the database. '''
        from sqlalchemy import MetaData, Table, Column
        from sqlalchemy import String, DateTime, ForeignKey

        metadata = MetaData()

        Table('master_cats', metadata,
            Column('id', String, primary_key = True),
            Column('name', String)
        )

        Table('cats', metadata,
            Column('id', String, primary_key = True),
            Column('name', String),
            Column('master_cat', String, ForeignKey('master_cats.id'))
        )

        Table('papers', metadata,
            Column('id', String, primary_key = True),
            Column('updated', DateTime),
            Column('published', DateTime),
            Column('title', String),
            Column('abstract', String)
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
        from tqdm.auto import tqdm

        master_cats = {
            'physics': 'Physics',
            'math': 'Mathematics',
            'cs': 'Computer Science',
            'q-bio': 'Quantitative Biology',
            'q-fin': 'Quantitative Finance',
            'stats': 'Statistics'
        }

        master_cat_query = 'insert or ignore into master_cats values '
        master_cat_query += ','.join(f'("{id}", "{name}")' 
                          for id, name in master_cats.items())

        ids, names, mcats = [], [], []
        base_url = 'http://arxitics.com/help/categories'
        for master_cat in tqdm(master_cats, desc = 'Setting up categories'):
            response = requests.get(base_url, {'group': master_cat})
            soup = BeautifulSoup(response._content, 'lxml')
            for li in soup.find_all('li'):
                if li.strong is not None:
                    ids.append(li.strong.text)
                    names.append(li.span.text[2:])
                    mcats.append(master_cat)

        cat_query = 'insert or ignore into cats values '
        cat_query += ','.join(f'("{id}", "{name}", "{mcat}")' 
                          for id, name, mcat in zip(ids, names, mcats))

        with self.engine.connect() as conn:
            conn.execute(master_cat_query)
            conn.execute(cat_query)

        return self

    def insert_papers(self, papers: list):
        ''' Insert papers into the database.
        
        INPUT
            papers: list
                A list of dictionaries, each containing the following keys:
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
                        The ArXiv categories that the paper falls under, 
                        separated by commas
        '''

        paper_query = 'insert or ignore into papers values '
        paper_query += ','.join(f'''
                       (     
                            "{paper['paper_id']}", 
                            "{paper['updated']}", 
                            "{paper['published']}",
                            "{paper['title']}", 
                            "{paper['abstract']}"
                        )'''
                       for paper in papers)

        paper_cat_query = 'insert or ignore into papers_cats values '
        paper_cat_query += ','.join(f'''
                     (
                         "{paper['paper_id']}", 
                         "{cat.strip()}"
                     )'''
                     for paper in papers 
                     for cat in paper['categories'].split(','))

        author_query = 'insert or ignore into authors values '
        author_query += ','.join(f'("{author.strip()}")'
                                 for paper in papers 
                                 for author in paper['authors'].split(','))

        paper_author_query = 'insert or ignore into papers_authors values '
        paper_author_query += ','.join(f'''
                     (
                         "{paper['paper_id']}", 
                         "{author.strip()}"
                     )'''
                     for paper in papers 
                     for author in paper['authors'].split(','))

        with self.engine.connect() as conn:
            conn.execute(paper_query)
            conn.execute(paper_cat_query)
            conn.execute(author_query)
            conn.execute(paper_author_query)

        return self

    def get_training_df(self):
        ''' Get a dataframe with ids, titles, abstracts and categories
            of all the papers in the database.

        OUTPUT
            A Pandas DataFrame object with columns id, title, abstract and
            a column for every ArXiv category
        '''
        import pandas as pd
        from tqdm.auto import tqdm

        with db.engine.connect() as conn:

            # Convert the papers table in the database to a dataframe
            df = pd.read_sql_table(
                table_name = 'papers', 
                con = conn, 
                columns = ['id', 'title', 'abstract'],
            )

            # Get a list of all the categories
            cat_result = conn.execute('select cats.id from cats')
            cats = [cat[0] for cat in cat_result]

            # Add every category as a column to the dataframe, with 0/1
            # values associated to each paper, signifying whether the paper
            # falls under that category
            for cat in tqdm(cats, desc = 'Creating dataframe'):
                query = f'''select paper_id, category_id from papers_cats
                            where category_id = "{cat}"'''
                paper_ids = [paper[0] for paper in conn.execute(query)]
                df[cat] = df['id'].isin(paper_ids).astype(int)

        return df

def clean(doc: str):
    ''' Clean a document. This removes newline symbols, scare quotes,
        superfluous whitespace and replaces equations with -EQN-. 
        
    INPUT
        doc: str
            A document

    OUTPUT
        The cleaned version of the document
    '''
    import re

    # Remove newline symbols
    doc = re.sub('\n', ' ', doc)

    # Convert LaTeX equations of the form $...$, $$...$$, \[...\]
    # or \(...\) to -EQN-
    dollareqn = '(?<!\$)\${1,2}(?!\$).*?(?<!\$)\${1,2}(?!\$)'
    bracketeqn = '\\[\[\(].*?\\[\]\)]'
    eqn = f'({dollareqn}|{bracketeqn})'
    doc = re.sub(eqn, ' -EQN- ', doc)

    # Remove scare quotes, both as " and \\"
    doc = re.sub('(\\"|")', '', doc)

    # Merge multiple spaces
    doc = re.sub(r' +', ' ', doc)

    return doc.strip()

def fetch(category: str, all_cats: list, max_results: int = 5, start: int = 0):
    ''' Fetch papers from the arXiv.

    INPUT
        category: str
            The name of the ArXiv category. Leave blank to search among all
            categories
        all_cats: list
            A list of all the categories
        max_results: int = 5
            Maximal number of papers scraped, ArXiv limits this to 10,000
        start: int = 0
            The index of the paper from which the scraping begins

    OUTPUT
        A list of dictionaries representing each paper entry, with each
        dictionary having the following keys:
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

    # Convert data formats and store it in a list
    papers = []
    for entry in soup.find_all('entry'):

        cats = ','.join(cat['term'] 
            for cat in entry.find_all('category')
            if cat['term'] in all_cats)

        if cats == '':
            continue

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

def scrape(db_name: str = 'arxiv_data.db', data_dir: str = 'data', 
    batch_size: int = 1000, patience: int = 20, overwrite: bool = False, 
    start_from: str = None):
    ''' Scrape papers from the ArXiv.

    INPUT
        db_name: str = 'arxiv_data.db'
            Name of the SQLite databse where the data will be stored
        data_dir: str = 'data'
            Directory in which the data files are to be stored
        batch_size: int = 1000
            The amount of papers fetched at each GET request - ArXiv limits
            this to 10,000
        patience: int = 20
            The amount of successive failed GET requests before moving on
            to the next category. The ArXiv API usually times out, resulting
            in a failed GET request, so this number should be reasonably
            large to rule these timeouts out
        overwrite: bool = False
            Whether the database file should be overwritten
        start_from: str = None
            A category to start from, which defaults to starting from scratch
    '''
    from time import sleep
    from tqdm.auto import tqdm
    from pathlib import Path

    # Create data directory
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        data_dir.mkdir()

    # Remove existing database if we are overwriting
    if overwrite:
        (data_dir / db_name).unlink()

    # Load database or create new one if it does not exist
    db = ArXivDatabase(name = db_name, data_dir = data_dir)

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
                    start = cat_idx,
                    all_cats = cats
                )

                # Wait for a couple of seconds to give the API a rest
                sleep(5)

                # Add a strike if there were no results, or reset the
                # strikes if there was a result
                if len(batch):
                    strikes = 0
                else:
                    strikes += 1
                    continue

                # Insert data into database
                db.insert_papers(batch)
                pbar.update(len(batch))
                cat_idx += len(batch)

if __name__ == '__main__':
    from pathlib import Path
    pcloud = Path.home() / 'pCloudDrive' / 'public_folder' / 'scholarly_data'

    db = ArXivDatabase(data_dir = pcloud)
    print(db.get_training_df())

    #scrape(data_dir = pcloud, start_from = 'astro-ph.SR')
