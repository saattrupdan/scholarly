import cleaner
import numpy as np
import pandas as pd
import os

def aggregate_cat(cat):
    if cat[:8] == 'astro-ph':
        agg_cat = 'physics'
    elif cat[:2] == 'cs':
        agg_cat = 'cs'
    elif cat[:5] == 'gr-qc':
        agg_cat = 'physics'
    elif cat[:3] == 'hep':
        agg_cat = 'physics'
    elif cat[:4] == 'math':
        agg_cat = 'math'
    elif cat[:4] == 'nlin':
        agg_cat = 'physics'
    elif cat[:4] == 'nucl':
        agg_cat = 'physics'
    elif cat[:7] == 'physics':
        agg_cat = 'physics'
    elif cat[:8] == 'quant-ph':
        agg_cat = 'physics'
    elif cat[:4] == 'stat':
        agg_cat = 'stats'
    else:
        agg_cat = 'other'
    return agg_cat

def aggregate_cats(cats):
    return np.asarray([aggregate_cat(cat) for cat in cats])

def agg_cats_to_binary(categories, agg_cats):
    '''
    Turns aggregated categories into a 0-1 sequence with 1's
    at every category index.
    
    INPUT
        categories, an iterable of strings
    
    OUTPUT
        numpy array with 1 at the category indexes and zeros everywhere else
    '''
    agg_categories = aggregate_cats(categories)
    return np.in1d(agg_cats, agg_categories).astype('int8')

def one_hot_agg(file_name, path = 'data'):
    ''' One-hot encode cats into binary aggregated cats. '''
    
    full_path = os.path.join(path, f"{file_name}_1hot_agg.csv")
    if os.path.isfile(full_path):
        print("File already one-hot encoded.")
    else:
        # get array of aggregated cats
        full_path = os.path.join(path, "cats.csv")
        agg_cats_df = pd.read_csv(full_path)
        agg_cats_df['category'] = agg_cats_df['category'].apply(aggregate_cat)
        agg_cats = np.asarray(agg_cats_df['category'].unique())

        # load cats data
        print("Loading cats...")
        full_path = os.path.join(path, f"{file_name}_cats.csv")
        clean_cats_with_path = lambda x: cleaner.clean_cats(x, path = path)
        cleaned_cats = pd.read_csv(
            full_path, 
            header = None, 
            converters = {0 : clean_cats_with_path}
        )
        print(f"Cats loaded!")
        
        # load ELMo data and merge with cats data
        print("Loading ELMo'd text...")
        full_path = os.path.join(path, f"{file_name}_elmo.csv")
        df_1hot_agg = pd.read_csv(full_path, header = None)
        df_1hot_agg['category'] = cleaned_cats.iloc[:, 0]
        df_1hot_agg = df_1hot_agg.dropna()
        print(f"ELMo data loaded!")
        
        # one-hot encode
        print("One-hot encoding...")
        bincat_arr = np.array([agg_cats_to_binary(cat_list, agg_cats)
            for cat_list in df_1hot_agg['category']]).transpose()
        bincat_dict = {key:value for (key,value) in zip(agg_cats, bincat_arr)}
        bincat_df = pd.DataFrame.from_dict(bincat_dict)
        print("One-hot encoding complete!")

        # add the one-hot encoded cats and drop the category column
        df_1hot_agg = pd.concat([df_1hot_agg, bincat_df], 
            axis = 1, sort = False)
        df_1hot_agg.drop(['category'], axis=1, inplace=True)

        # save the one-hot encoded dataframe
        full_path = os.path.join(path, f"{file_name}_1hot_agg.csv")
        df_1hot_agg.to_csv(full_path, index = False)

        print(f"Saved the one-hot encoded dataframe.")
