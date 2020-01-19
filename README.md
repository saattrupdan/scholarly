# Scholarly
Category classification of scientific papers. Given a title and an abstract of a paper, the model will predict a list of categories to which the paper belongs. These categories are the 148 categories used on [arXiv](https://arxiv.org).

## Usage
Go to [saattrupdan.pythonanywhere.com/scholarly](saattrupdan.pythonanywhere.com/scholarly) to test the model out. Note that it also supports LaTeX like $\frac{1}{5}$.

A REST API is also available at the [saattrupdan.pythonanywhere.com/scholarly/result](https://saattrupdan.pythonanywhere.com/scholarly/result) endpoint, with arguments `title` and `abstract`. You will then receive a JSON response containing a list of lists, with each inner list containing the category id, category description and the probability. The list will only include results with probabilities at least 50%, and the list is sorted descending by probability. [Here](https://saattrupdan.pythonanywhere.com/scholarly/result?title="test"&abstract="test") is an example of a query.

## Documentation and data
This model was trained on all titles and abstracts from all of [arXiv](https://arxiv.org) up to and including 2019, which were all scraped from their API. The scraping script can be found in `arxiv_scraper.py`. All the data can be found at

<p align=center>
  <a href="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data">
    https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data
  </a>
</p>


The main data file is the SQLite file `arxiv_data.db`, which contains 6 tables:
  - `cats`, containing the 148 arXiv categories
  - `master_cats`, containing the arXiv "master categories", which are 6 aggregates of the categories into things like Mathematics and Physics
  - `papers`, containing the id, date, title and abstract of papers
  - `papers_cats`, which links papers to their categories
  - `authors`, containing author names
  - `papers_authors`, which links authors to their papers

From this database I extracted the dataset `arxiv_data.tsv`, which contains the title and abstract for every paper in the database, along with a binary column for every category, denoting whether a paper belongs to that category. LaTeX equations in titles and abstracts have been replaced by "-EQN-" at this stage. Everything related to the database can be found in the `db.py` script.

A preprocessed dataset is also available, `arxiv_data_pp.tsv`, in which the titles and abstracts have been merged as "-TITLE_START- {title} -TITLE_END- -ABSTRACT_START- {abstract} -ABSTRACT_END-", and the texts have been tokenised using the SpaCy en_core_web_sm tokeniser. The resulting texts have all tokens separated by spaces, so that simply splitting the texts by whitespace will yield the tokenised versions. The preprocessing is done in the `data.py` script.

Two JSON files, `cats.json` and `mcat_dict.json` are also available, which are basically the `cats` table from the database and a dictionary to convert from a category to its master category, respectively.

I trained FastText vectors on the entire corpus, and the resulting model can be found as `fasttext_model.bin`, with the vectors themselves belonging to the text file `fasttext`. The script I used to train these can be found in `train_fasttext.py`.

In case you're like me and are having trouble working with the entire dataset, there's also the `arxiv_data_mini_pp.tsv` dataset, which simply consists of 100,000 randomly samples papers from `arxiv_data_pp.tsv`. You can make your own versions of these using the `make_mini.py` script, which constructs the smaller datasets without loading the larger one into memory.

The model itself is a simplified version of the new [SHA-RNN model](https://arxiv.org/abs/1911.11423), trained from scratch on the [BlueCrystal Phase 4 compute cluster](https://www.acrc.bris.ac.uk/acrc/phase4.htm) at the University of Bristol, UK. All scripts pertaining to the model are `modules.py`, `training.py` and `inference.py`.

## Contact
If you have any questions regarding the data or model, please contact me at saattrupdan at gmail dot com.
