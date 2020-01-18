import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from tqdm.auto import tqdm

def predict(model, title: str, abstract: str):
    ''' Get the predicted categories from a model, a title and an abstract.

    INPUT
        model: torch.nn.Module
            A trained model
        title: str
            The title of a research paper
        abstract: str
            The abstract of a research paper
    
    OUTPUT
        A list of pairs (category_name, probability), with category_name
        being the official name of the arXiv category, such as math.LO,
        and the probability being the outputted sigmoid value from the
        model. The list will only contain pairs where the probability is
        above 50%.
    '''
    import spacy
    from utils import get_cats

    # Merge title and abstract
    text = f'-TITLE_START- {title} -TITLE_END- '\
           f'-ABSTRACT_START- {abstract} -ABSTRACT_END-'

    # Load the tokeniser
    nlp = spacy.load('en')
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    # Numericalise the tokens
    idxs = torch.LongTensor([[model.stoi[tok] for tok in tokenizer(text)]])

    # Get predictions
    logits = model(idxs.transpose(0, 1))
    probs = torch.sigmoid(logits)

    # Get the categories corresponding to the predictions
    cats = get_cats(data_dir = model.data_dir)
    sorted_idxs = probs.argsort(descending = True)
    predicted_cats = [(cats[idx], round(float(probs[idx]), 2))
        for idx in sorted_idxs if probs[idx] > 0.5]

    return predicted_cats 

def evaluate(model, val_dl, output_dict: bool = False):
    ''' Evaluate a model on a validation dataset.

    INPUT
        model: torch.nn.Module
            A trained model
        val_dl: torch.utils.data.DataLoader
            A data loader containing a validation dataset
        output_dict: bool = False
            Whether to output a dictionary instead of a string

    OUTPUT
        A string or dictionary containing a classification report
    '''
    from sklearn.metrics import classification_report
    import warnings
    from utils import get_cats

    with torch.no_grad():
        model.eval()

        y_vals, y_hats = [], []
        for x_val, y_val in val_dl:

            if model.is_cuda():
                x_val = x_val.cuda()
                y_val = y_val.cuda()

            yhat = model(x_val)
            preds = torch.sigmoid(yhat) > 0.5

            y_vals.append(y_val.int())
            y_hats.append(preds.int())

        y_val = torch.cat(y_vals, dim = 0)
        y_hat = torch.cat(y_hats, dim = 0)

        cats = get_cats(data_dir = model.data_dir)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            report = classification_report(
                y_true = y_val.cpu(), 
                y_pred = y_hat.cpu(), 
                target_names = cats,
                output_dict = output_dict
            )
        return report


if __name__ == '__main__':
    pass
