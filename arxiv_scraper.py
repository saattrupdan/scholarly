from urllib import request
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime as dt
import numpy as np
from time import sleep

def fetch(category="", max_results=5, start=0):
    """ Fetch papers from the arXiv.

    INPUT
        category="":    str, arXiv category
        max_results=5:  int, maximal number of papers scraped (arXiv limits this to 30,000)
        start=0:        int, from which paper the scraping begins

    OUTPUT
        pandas DataFrame object with ids, titles, authors, abstracts, categories, pdf links, updated date and published date."""

    # the url needs a colon if only if the category string is non-empty
    if category:
        category = ":" + category

    # fetch data and create soup
    url = f'http://export.arxiv.org/api/query?search_query=cat{category}&start={start}&max_results={max_results}&sortBy=lastUpdatedDate&sortOrder=descending'
    with request.urlopen(url) as url_data:
        soup = BeautifulSoup(url_data, "xml")

    # fetch paper data from soup
    papers = {'id' : [], 'title' : [], 'authors' : [], 'abstract' : [], 'category' : [], 'pdf' : [], 'updated' : [], 'published' : []}
    for entry in soup.find_all(name="entry"):
        
        # save metadata
        try:
            entry_id = entry.id.string
            entry_updated = dt.strptime(entry.updated.string[:10], "%Y-%m-%d")
            entry_published = dt.strptime(entry.published.string[:10],"%Y-%m-%d")
            entry_title = entry.title.string.strip("\n")
            entry_abstract = entry.summary.string.strip("\n")

            # collect all authors in a list 
            entry_authors = []
            for author in entry.find_all(name="author"):
                for name in author.find_all(name="name"):
                    entry_authors.append(name.string)

            if entry_authors == []:
                entry_authors = np.nan

            # try to store category, and store NaN if impossible
            entry_cats = []
            for cat in entry.find_all(name="category"):
                try:
                    entry_cats.append(cat['term'])
                except:
                    pass

            if entry_cats == []:
                entry_cats = np.nan


            # try to store pdf link, and store NaN if impossible
            entry_pdf = np.nan
            for link in entry.find_all(name="link"):
                try:
                    if link['title'] == 'pdf':
                        entry_pdf = link['href']
                except:
                    pass

        except:
            continue
        
        # store data into dictionary
        papers['id'].append(entry_id)
        papers['updated'].append(entry_updated)
        papers['published'].append(entry_published)
        papers['title'].append(entry_title)
        papers['authors'].append(entry_authors)
        papers['abstract'].append(entry_abstract)
        papers['category'].append(entry_cats)
        papers['pdf'].append(entry_pdf)
        

    # create dataframe from papers
    return pd.DataFrame(papers)




def batch_fetch(category="", max_results=5, file_path="paper_data", start=0, batch_size=1000):
    """Fetch papers from the arXiv in batches and store them in a .csv file.
    
    INPUT
        category="":                str, arXiv category
        max_results=5:              int, maximal number of papers scraped
        file_path="paper_data":     str, csv file path, can include file extension or not
        start=0:                    int, from which paper the scraping begins
        batch_size=1000:            int, how many papers are loaded at a time (arXiv limits this to 30,000)

    OUTPUT
        nothing """
    
    # remove file extension in file_path if it was included
    if file_path[-4:] == ".csv":
        file_path = file_path[:-4]
    
    # calculate how many batches there are
    if max_results % batch_size == 0:
        batches = max_results // batch_size
    else:
        batches = max_results // batch_size + 1
    
    # for every batch download data into a separate temporary .csv file
    for i in range(batches):
        
        if max_results % batch_size == 0 or i != batches - 1:
            result_size = batch_size
        else:
            result_size = max_results % batch_size 

        # make a loop to keep trying to get data in case arxiv spits out nothing for some reason 
        df = fetch(category, result_size, i * batch_size + start)
        time_waited = 0
        while df.size == 0: 
            sleep(30)
            time_waited += 30
            if time_waited == 120:
                print(f"BATCH {i+1}/{batches}: Timed out. Might have scraped everything in {category}.")
                break

            df = fetch(category, result_size, i * batch_size + start)
        
        if df.size != 0:
            df.to_csv(f"{file_path}_{i}.csv", index=False)
            print(f"BATCH {i+1}/{batches}: Success! Loaded {result_size} paper(s) into the temporary file {file_path}_{i}.csv.")
        else:
            batches = i
            break

    # merge all .csv files
    df = pd.concat( [ pd.read_csv(f"{file_path}_{i}.csv") for i in range(batches) ] )
    df.to_csv(f"{file_path}.csv", index=False)
    print(f"Merged {batches} temporary file(s) into {file_path}.csv.")
    
    # remove the temporary files
    for i in range(batches):
        os.remove(f"{file_path}_{i}.csv")




def cat_scrape(max_results_per_cat=10000, file_path="arxiv_data", batch_size=100, start_cat="astro-ph"):
    
    all_cats = list(pd.read_csv("cats.csv")['category'])
    start_index = all_cats.index(start_cat)
    cats = all_cats[start_index:]
    
    for i, cat in enumerate(cats):
        print(f"CATEGORY {i+1}/{len(cats)}: Scraping {max_results_per_cat} papers from {cat}...")
        batch_fetch(category=cat, max_results=max_results_per_cat, file_path=f"{file_path}_{cat}", start=0, batch_size=batch_size)
        print("")
    
    # merge all .csv files
    df = pd.concat( [ pd.read_csv(f"{file_path}_{cat}.csv") for cat in all_cats ] )
    df.to_csv(f"{file_path}.csv", index=False)

    # remove the temporary files
    for cat in all_cats:
        os.remove(f"{file_path}_{cat}.csv")

    print(f"Scraped {max_results_per_cat * len(cats)} papers and loaded them into {file_path}.csv.")
