class ArXivDatabase:
    ''' A SQLite databse for storing ArXiv papers. 
    
    INPUT
        name: str = 'arxiv_data.db'
            Name of the database
        data_dir: str = '.data'
            Folder which contains the database
    '''

    def __init__(self, name: str = 'arxiv_data.db', data_dir: str = '.data'):
        from sqlalchemy import create_engine
        from utils import get_path
        db_path = get_path(data_dir) / name
        self.engine = create_engine(f'sqlite:///{db_path}')
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

        p_entry = lambda paper:\
            f'''
            (     
                "{paper['paper_id']}", 
                "{paper['updated']}", 
                "{paper['published']}",
                "{paper['title']}", 
                "{paper['abstract']}"
            )
            '''
        a_entry = lambda author:\
            f'''
            (
                "{author.strip()}"
            )
            '''
        pc_entry = lambda paper, cat:\
            f'''
            (
                "{paper['paper_id']}", 
                "{cat.strip()}"
            )
            '''
        pa_entry = lambda paper, author:\
            f'''
            (
                "{paper['paper_id']}", 
                "{author.strip()}"
            )
            '''

        p_query = 'insert or ignore into papers values '
        p_query += ','.join(p_entry(paper) for paper in papers)

        a_query = 'insert or ignore into authors values '
        a_query += ','.join(a_entry(author) for paper in papers 
                            for author in paper['authors'].split(','))

        pc_query = 'insert or ignore into papers_cats values '
        pc_query += ','.join(pc_entry(paper, cat) for paper in papers 
                             for cat in paper['categories'].split(','))

        pa_query = 'insert or ignore into papers_authors values '
        pa_query += ','.join(pa_entry(paper, author) for paper in papers 
                             for author in paper['authors'].split(','))

        with self.engine.connect() as conn:
            conn.execute(p_query)
            conn.execute(a_query)
            conn.execute(pc_query)
            conn.execute(pa_query)

        return self

    def get_cats(self, conn = None, data_dir: str = '.data') -> list:
        ''' Get a list of all the categories. '''
        from utils import get_path
        import json

        if conn is None:
            with self.engine.connect() as conn:
                cat_result = conn.execute('select id from cats')
                cats = [cat[0] for cat in cat_result]
        else:
            cat_result = conn.execute('select id from cats')
            cats = [cat[0] for cat in cat_result]

        if save_to is not None:
            with open(get_path(data_dir) / 'cats.json', 'w') as f:
                json.dump(cats, f)

    def get_mcat_dict(self, conn = None, data_dir: str = '.data') -> dict:
        ''' Get a dictionary mapping each category to its master category. '''
        from utils import get_path
        import json

        if conn is None:
            with self.engine.connect() as conn:
                mcat_result = conn.execute('select id, master_cat from cats')
                mcat_dict = {pair[0]: pair[1] for pair in mcat_result}
        else:
            mcat_result = conn.execute('select id, master_cat from cats')
            mcat_dict = {pair[0]: pair[1] for pair in mcat_result}

        if save_to is not None:
            with open(get_path(data_dir) / 'mcat_dict.json', 'w') as f:
                json.dump(mcat_dict, f)

        return mcat_dict

    def get_training_df(self, only_master_cats: bool = False):
        ''' Get a dataframe with titles, abstracts and categories
            of all the papers in the database.

        INPUT
            only_master_cats: bool = False
                Whether to have columns for every master category, or to
                have columns for every category

        OUTPUT
            A Pandas DataFrame object with columns title, abstract (or a
            single text column if merge_title_abstract is True) and
            a column for every ArXiv category or master category
        '''
        import pandas as pd
        from tqdm.auto import tqdm

        with self.engine.connect() as conn:

            cats = self.get_cats(conn)
            mcat_dict = self.get_mcat_dict(conn)

            # Convert the papers table in the database to a dataframe
            df = pd.read_sql_table(
                table_name = 'papers', 
                con = conn, 
                columns = ['id', 'title', 'abstract'],
            )

            # Add every category as a column to the dataframe, with 0/1
            # values associated to each paper, signifying whether the paper
            # falls under that category
            for cat in tqdm(cats, desc = 'Creating dataframe'):
                query = f'''select paper_id, category_id from papers_cats
                            where category_id = "{cat}"'''
                paper_ids = [paper[0] for paper in conn.execute(query)]

                bool_col = df['id'].isin(paper_ids).astype(int)
                if only_master_cats:
                    mcat = mcat_dict[cat]
                    if mcat in df.columns:
                        df[mcat] = (df[mcat] | bool_col).astype(int)
                    else:
                        df[mcat] = bool_col
                else:
                    df[cat] = bool_col

        return df.drop(columns = ['id'])

if __name__ == '__main__':
    from utils import get_path

    # Open database
    db = ArXivDatabase(data_dir = '.data')

    db.get_mcat_dict(save_to = get_path('.data') / 'mcat_dict.json')
    db.get_cats(save_to = get_path('.data') / 'cats.json')

    # Pull out a abstracts and categories into a tsv file
    #df = db.get_training_df(only_master_cats = True)
    #df.to_csv(get_path('.data') / 'arxiv_data_mcats.tsv', sep = '\t', 
    #    index = False)

    # Example query: Output the number of authors in database
    #with db.engine.connect() as conn:
    #    query = 'select * from authors'
    #    authors = [author[0] for author in conn.execute(query)]
    #    print(len(authors))
