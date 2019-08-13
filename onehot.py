import cleaner
import numpy as np
import pandas as pd
import os
import itertools as it
import shutil

def aggregate_cat(cat):
    if cat[:8] == 'astro-ph':
        agg_cat = 'physics'

    elif cat[:2] == 'cs':
        agg_cat = 'computer science'

    elif cat[:5] == 'gr-qc':
        agg_cat = 'physics'

    elif cat[:3] == 'hep':
        agg_cat = 'physics'

    elif cat[:4] == 'math':
        agg_cat = 'mathematics'

    elif cat[:4] == 'nlin':
        agg_cat = 'physics'

    elif cat[:4] == 'nucl':
        agg_cat = 'physics'

    elif cat[:7] == 'physics':
        agg_cat = 'physics'

    elif cat[:8] == 'quant-ph':
        agg_cat = 'physics'

    elif cat[:4] == 'stat':
        agg_cat = 'statistics'

    elif cat[:4] == 'econ':
        agg_cat = 'statistics'

    elif cat[:4] == 'eess':
        agg_cat = 'electrical engineering and systems science'
    
    elif cat[:8] == 'cond-mat':
        agg_cat = 'physics'
    
    elif cat[:5] == 'q-bio':
        agg_cat = 'biology'
    
    elif cat[:5] == 'q-fin':
        agg_cat = 'finance'

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

def cats_to_binary(cat_list, all_cats):
    '''
    Turns categories into a 0-1 sequence with 1's
    at every category index.
    
    INPUT
        categories, an iterable of strings
    
    OUTPUT
        numpy array with 1 at the category indexes and zeros everywhere else
    '''
    return np.in1d(all_cats, cat_list).astype('int8')


def one_hot(file_name, path = 'data', batch_size = 256, onehot_name = '1hot'):
    ''' One-hot encode cats into binary cats. '''

    print(f"One-hot encoding {file_name}_{onehot_name}...")

    # set function that creates binary cats
    if onehot_name == '1hot':
        bin_fn = cats_to_binary
    elif onehot_name == '1hot_agg':
        bin_fn = agg_cats_to_binary

    full_path = os.path.join(path, f"{file_name}_{onehot_name}.csv")
    if os.path.isfile(full_path):
        print("File already has one-hot encoded cats.")
    else:
        # get array of cats
        full_path = os.path.join(path, "cats.csv")
        cats_df = pd.read_csv(full_path)

        if onehot_name == '1hot_agg':
            cats_df['category'] = cats_df['category'].apply(aggregate_cat)
        
        all_cats = np.asarray(cats_df['category'].unique())

        clean_path = os.path.join(path, f"{file_name}_clean.csv")
        clean_cats_with_path = lambda x: cleaner.clean_cats(x, path = path)

        temp_dir = os.path.join(path, f'{file_name}_{onehot_name}_temp')
        if not os.path.isdir(temp_dir):
            os.system(f'mkdir {temp_dir}')
        
        clean_batches = pd.read_csv(
            clean_path, 
            header = None, 
            converters = {0 : clean_cats_with_path},
            chunksize = batch_size
            )

        for (i, batch) in enumerate(clean_batches):
            
            # reset index of batch, which will enable
            # concatenation with other dataframes
            batch.reset_index(inplace = True, drop = True)

            # if file already exists then skip it
            full_path = os.path.join(temp_dir,
                f'{file_name}_{onehot_name}_{i}_csv')
            if os.path.isfile(full_path):
                continue

            # one-hot encode
            bincat_arr = np.array([bin_fn(cat_list, all_cats)
                for cat_list in batch.iloc[:, 0]]).transpose()
            bincat_dict = {key:value for (key,value) in
                zip(all_cats, bincat_arr)}
            df_1hot = pd.DataFrame.from_dict(bincat_dict)

            # merge clean df and one-hot cats
            df_1hot = pd.concat(
                [batch.iloc[:, 1], df_1hot], 
                axis = 1, 
                )

            # remove rows with no categories
            no_cats = np.argwhere(
                        np.equal(
                            np.sum(bincat_arr, axis = 0),
                            0
                            )
                        ).T.squeeze()
            df_1hot.drop(index = no_cats)
            
            # remove rows with invalid text
            df_1hot.dropna(inplace = True)

            # save the one-hot encoded dataframe
            full_path = os.path.join(temp_dir,
                f"{file_name}_{onehot_name}_{i}.csv")
            df_1hot.to_csv(full_path, index = False, header = False)

            print(f'One-hot encoded {(i+1) * batch_size} papers...',
                end = '\r')

        print("One-hot encoding complete!" + " " * 25)
        print("Merging temporary files...")

        full_path = os.path.join(path, f'{file_name}_{onehot_name}.csv')
        with open(full_path, 'wb+') as file_out:
            for i in it.count():
                print(f"{i} files merged...", end = "\r")
                try:
                    full_path = os.path.join(temp_dir,
                        f"{file_name}_{onehot_name}_{i}.csv")
                    with open(full_path, "rb") as file_in:
                        shutil.copyfileobj(file_in, file_out)
                except IOError:
                    break
            print("Merge complete!" + " " * 25)

        print("Removing temporary files...")
        shutil.rmtree(temp_dir)
        try:
            os.rmdir(temp_dir)
        except:
            pass
        print("Removal complete!")
