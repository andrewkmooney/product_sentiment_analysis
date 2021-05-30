import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk import FreqDist
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import svm
import matplotlib.pyplot as plt
import spacy
import time
import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load("en_core_web_sm")

stopwords_list = stopwords.words('english') + [' ', '  ', '   ', 'w/']

def process_text(text, is_spacy=False, keep_stopwords=True, keep_links=False, keep_hashtags = True):
    
    '''
    Processes, cleans, and tokenizes text with multiple options on how to do so. Returns either a SpaCy doc object or a list of tokens
    
    Parameters
    ----------
    text - (str) - the text to be processed
    is_spacy - (bool) - whether or not the return is a SpaCy doc with stop words included. If True is passed, keep_stopwords will be ignored
    keep_stopwords - (bool) - whether or not to remove stopwords using the nltk stopwords library
    keep_links - (bool) - whether or not to turn a handle and website link into a blank word such as WEBSITE for the purposes of sentence structure vectorization
    keep_hashtags - (bool) - whether or not to simply remove the # symbol from a hashtag or remove the hashtag completely
    
    Returns
    -------
    Processed text - doc, if OR list
    
    '''
    
    if keep_links == True: # Removes all {links} as well as urls and @handles which sometimes include a preceeding period
        text = re.sub('\.?(@[A-Za-z0-9_]+)', 'HANDLE', text)
        text = re.sub('{([A-Za-z]+)}', 'WEBSITE', text)
        text = re.sub('(http://[A-Za-z./0-9?=-]+)', 'WEBSITE', text)
    else:
        text = re.sub('\.?(@[A-Za-z0-9_]+)', '', text)
        text = re.sub('{([A-Za-z]+)}', '', text)
        text = re.sub('(http://[A-Za-z./0-9?=-]+)', '', text)
    
    if keep_hashtags == False: # Either removes the complete hashtag or just the '#' symbol
        text = re.sub('(#[A-Za-z0-9_]+)', '', text)
    else:
        text = text.replace('#', '')
        
    text = re.sub('&([A-Za-z])+;', '', text) # Removes special html characters such as &quot
    text = re.sub('(\s){2,}', ' ', text) # Turns double/triple spaces into single spaces
    text = text.replace('RT', '') # Removes if the tweets have been retweeted
    text = text.replace('\x89Û÷', '') # Removes special characters
    text = text.replace('\x89Ûª', '')
    text = text.replace('\x89ÛÏ', '')
    text = text.replace('\x9d', '')
    text = text.replace('\x89Û', '')
    text = text.replace('\x89ÛÒ', '')
    text = text.replace('[pic]', '') # Removes picture links
    text = text.replace('(\\x89)(\S)*', '')
    doc = nlp(text) # Turns the resulting string into a SpaCy doc object
    
    if is_spacy == True:
        return doc
    
    text = text.replace("'", '') # Removes apostrophes
    
    if keep_stopwords == False: # Makes all tokens lower case, removes punctuation and stopwords
        return [x.lower() for x in list([token.text for token in doc if token.is_punct == False and token.text.lower() not in stopwords_list])]
    else: # Does the same thing, but keeps stopwords in
        return [x.lower() for x in list([token.text for token in doc if token.is_punct == False])]

class TextSet:

    
    def __init__(self, X, y, name=None, is_spacy=False, keep_stopwords=False, keep_links=False, keep_hashtags=False, random_seed=42, split=.3):
        '''

        Constructs necessary attributes for the Model_Analysis object and automatically processes the X data into Holdout, 
        Train and Test sets ready for vectorization. Also proceses y into labels and sparse OHE matrices, split into Holdout, 
        Train, and Test sets. The Holdout set is 10% of the original test data.

        Parameters:
        ----------

        X : Series or Array
            The text that will be processed in a fashion determined by the other arguments. It will be split into holdout, train, and test batches for testing. 
        y : Series or Array
            The target values for the accompanying texts passed in X. Will be processed with both a label encoder and a one hot encoded sparse matrix. Must be the same dimensions as X.
        name : String
            The name of the TextSet. Used for labeling purposes during analysis.
        is_spacy : Boolean
            Whether or not the text will be processed into a SpaCy doc and will use the SpaCy vectorization method.
        keep_stopwords: Boolean
            Whether or not the processed text will include stop words
        keep_links: Boolean
            If True is passed, Twitter handles and Web URLs will be replaced with dummy words. If False, then the links will be removed completely
        keep_hashtags: Boolean
            If True is passed, the hashtag symbol will be removed, but the rest of the hashtag will remain. If False, then the entire hashtag will be removed.
        random_seed: int
            The random state used for partitioning data
        split: float
            The size of the test set when splitting data into train and test

        '''

        self.X = X
        self.y = y
        self.name = name
        self.is_spacy = is_spacy
        self.keep_stopwords = keep_stopwords
        self.keep_links = keep_links
        self.keep_hashtags = keep_hashtags
        self.random_seed = random_seed
        self.split = split
        self.process_data()
            
    def process_data(self):
        '''
        Takes X and y and tokenizes the text based on the methods determined by the class's parameters. 
        The data is then split into a 10% holdout group, train, and test groups. The target y variable is 
        processed into labels and one hot encoded sparse matricies.
        
        Parameters
        ----------
        
        None
        
        Returns
        -------
        text : List
            A list of all processed tokens in the X column
        X_holdout_tokens : List
            Tokenized list of the X holdout group
        X_train_tokens : List
            Tokenized list of the X train group
        X_test_tokens : List
            Tokenized list of the X test group
        y_holdout_ohe : Array
            A sparse matrix of the y holdout group
        y_holdout_l : Array
            A 1D labeled array of the y train group
        y_train_ohe : Array
            A sparse matrix of the y train group
        y_train_l : Array
            A 1D labeled array of the y train group
        y_test_ohe : Array
            A sparse matrix of the y test group
        y_test_l : Array
            A 1D labeled array of the y test group
        y_ohe : OneHotEncoder
            The one hot encoder used to transform y values into sparse matricies
        y_labeler : LabelEncoder
            The label encoder used to transform the y values into the labels fo testing
        process_time 
        : int
            Amount of time in seconds for text to process
        '''
        start_time = time.time()
        
        self.tokens = [process_text(tweet, 
                                    is_spacy=self.is_spacy, 
                                    keep_stopwords=self.keep_stopwords, 
                                    keep_links=self.keep_links, 
                                    keep_hashtags=self.keep_hashtags) for tweet in self.X]
        
        
        X_processed, self.X_holdout_tokens, y_processed, self.y_holdout = train_test_split(self.tokens, 
                                                                              self.y, 
                                                                              random_state=self.random_seed, 
                                                                              test_size=.1)
        
        self.X_train_tokens, self.X_test_tokens, self.y_train, self.y_test = train_test_split(X_processed, 
                                                                              y_processed, 
                                                                              random_state=self.random_seed, 
                                                                              test_size=self.split)
        
        if self.is_spacy == True:
            token_list = []
            for doc in self.tokens:
                token_list.append([token.text for token in doc])
            self.tokens = token_list
            
        self.text = [' '.join(x) for x in self.tokens]
        
        encoder = OneHotEncoder()
        self.y_train_ohe = encoder.fit_transform(np.array(self.y_train).reshape(-1,1))
        self.y_test_ohe = encoder.transform(np.array(self.y_test).reshape(-1,1))
        self.y_holdout_ohe = encoder.transform(np.array(self.y_holdout).reshape(-1,1))
        self.y_ohe = encoder
        
        labeler = LabelEncoder()
        self.y_train_l = labeler.fit_transform(self.y_train)
        self.y_test_l = labeler.transform(self.y_test)
        self.y_holdout_l = labeler.transform(self.y_holdout)
        self.y_labeler = labeler
        
        self.y_train_ohe = self.y_train_ohe.toarray()
        self.y_test_ohe = self.y_test_ohe.toarray()
        self.y_holdout_ohe = self.y_holdout_ohe.toarray()
        self.process_time = time.time() - start_time
        
        print("--- %s seconds ---" % (self.process_time))
        
        
    def word_cloud(self):
        
        '''
        Visualizes the entire corpus of X into a frequency wordcloud
        
        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        wordcloud : WordCloud
            A visualization of the wordcloud on a (6,6) graph
        '''

        words = ' '.join(self.text)

        self.wordcloud = WordCloud(width = 600, height = 600,
                        background_color ='white',
                        min_font_size = 10).generate(words)

        # plot the WordCloud image                       
        plt.figure(figsize = (6, 6), facecolor = None)
        plt.imshow(self.wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.title(f'Word Cloud For {self.name}')
        plt.show()
        
    
    def plot_frequency(self, num_words = 50):
        
        '''
        Creates a bar graph of the frequencies of each token in the complete corpus of X
        
        Parameters
        ----------
        
        num_words : int
            Number of words to be displayed
            
        Returns
        -------
        
        Bar Plot : graph
            A bar plot of the most frequent words in the X corpus
        '''
    
        data_concat = []

        for tweet in self.tokens:
            data_concat += tweet

        data_freqdist = FreqDist(data_concat)

        x = []
        y = []

        for token in data_freqdist.most_common(num_words):
            x.append(token[0])
            y.append(token[1])

        plt.figure(figsize=(15, 6))
        plt.bar(x=x, height=y)
        plt.xticks(rotation=45)
        plt.xlabel('Words')
        plt.title(f'{self.name} {num_words} Most Common Words')
        plt.ylabel('Frequency')
        plt.show()
    
    def vectorize(self, method='tf_idf', max_features=300, ngram_range=(1,1)):
        
        '''
        Turns X_train, X_test, and X_holdout into vectors for processing 
        
        Parameters
        ----------
        
        method : String - 'tf_idf' or 'count'
            Method of vectorization if is_spacy is False
        max_features : int
            Number of words to be included in the count or TF-IDF vectorization
        ngram_range : tuple
            Number of words to be considered as paired for tokenization
        
        Returns
        -------
        
        X_train: Array
            A vector representation of X_train tokens
        X_test: Array
            A vector representation of X_test tokens
        X_holdout: Array
            A vector representation of X_holdout tokens
        vectorizer: CountVectorizer or TfidfVectorizer
            The vectorizer used to transform the X data
        
        '''
        
        if self.is_spacy == True:
            self.X_train = np.array([doc.vector for doc in self.X_train_tokens])
            self.X_test = np.array([doc.vector for doc in self.X_test_tokens])
            self.X_holdout = np.array([doc.vector for doc in self.X_holdout_tokens])
        
        else:
            X_train = [' '.join(x) for x in self.X_train_tokens]
            X_test = [' '.join(x) for x in self.X_test_tokens]
            X_holdout = [' '.join(x) for x in self.X_holdout_tokens]
            
            if method == 'tf_idf':
                self.vectorizor = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range)
                self.X_train = self.vectorizor.fit_transform(X_train)
                self.X_test = self.vectorizor.transform(X_test)
                self.X_holdout = self.vectorizor.transform(X_holdout)
                self.X_train = self.X_train.toarray()
                self.X_test = self.X_test.toarray()
                self.X_holdout = self.X_holdout.toarray()

            elif method == 'count':
                self.vectorizor = CountVectorizer(max_features=max_features, ngram_range=(1,1))
                self.X_train = self.vectorizor.fit_transform(X_train)
                self.X_test = self.vectorizor.transform(X_test)
                self.X_holdout = self.vectorizor.transform(X_holdout)
                self.X_train = self.X_train.toarray()
                self.X_test = self.X_test.toarray()
                self.X_holdout = self.X_holdout.toarray()
                
        
    def regularize(self):
        
        '''
        Normalizes the X vectors using a StandardScaler. Ideal for usage with neural networks. Must be run after the vectorize function.
        
        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        scaler: StandardScaler
            The scaler used to transform X vectors
        X_train_scaled: Array
            The scaled vector of X_train
        X_test_scaled: Array
            The scaled vector of X_test
        X_holdout_scaled: Array
            The scaled vector of X_holdout
        '''
            
        self.scaler = StandardScaler()
            
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.X_holdout_scaled = self.scaler.transform(self.X_holdout)

class ModelComparison:
    
    def __init__(self, pipeline, data_list, y_format='label', nn=False, name=None):
            '''

            Constructs necessary attributes for the ModelComparison object. If nn is True then y_format will be changed to 'ohe'.

            Parameters:
            ----------

            pipeline : Classifier
                An untrained classification alogorithm for comparing against data. A pipeline is recommended in order to control scaling and imputing, but a regular model can be passed.
            data_list : List
                A list of TextSet objects with X, y train, test and holdout sets already processed and vectorized
            y_format : 'label', 'ohe' or 'raw'
                The format of the y_data. If 'label' is passed, then y_train_l will be used. If 'ohe', then y_train_ohe will be used. Else, y_train tokens will be used.
            nn : Boolean
                Whether or not the pipeline is a neural network
            name : String
                The name of the ModelComparison Object.

            '''
            self.pipeline = pipeline
            self.data_list = data_list
            self.name = name
            self.y_format = y_format
            self.data_names = {j:i for (i, j) in tuple(enumerate([x.name for x in self.data_list]))}
            self.ref_dict = {dataset.name : dataset for dataset in self.data_list}
            self.nn = nn
            
            if self.nn == True:
                self.y_format = 'ohe'
    
    def set_model(self, new_model, y_change=None, nn_change=False):
        '''
        Changes the model used for comparison.
        
        Parameters
        ----------
        
        new_model : Classifier
            An untrained classification alogorithm for comparing against data.
        y_change : 'label', 'ohe', 'raw' or None
            The new configuration of the y variable
        nn_change : Boolean
            Pass True if the new model is a neural network, else pass False
        
        '''
        self.pipeline = new_model
        if y_change != None:
            self.y_format = y_change
        
        if nn_change == True:
            self.nn = nn_change

    def fit_models(self, data=None, batch_size=15, epochs=20):
        
        '''
        Fits the model to each of the datasets and produces data for comparison and visualization.
        
        Parameters
        ----------
        
        data : List
            A list of processed TextSet objects. If None is passed, then the model will train on every model in the ModelComparison object
        batch_size : int
            Size of batches used when training a Neural Network model. Ignored if nn is False.
        epochs : int
            Number of epochs used to train a Neural Network model. Ignored if nn is False.
            
        Returns
        -------
        
        data_dict : dictionary
            A dictionary with keys corresponding to the names of each dataset fit to the model. Contains all predicted y values for each dataset for visualization and comparison.
        score_comparison: dataframe
            A dataframe of scores for each dataframe with the maximum value in each row highlighted
        process_time: int
            Amount of time taken to fit the model to all datasets.
        '''
        
        start_time = time.time()
        self.data_dict = {}
        
        if data == None:
            set_list = self.data_list
        
        else:
            set_list = data
        
        for dataset in set_list:
            
            y_dict_train = {
                'label': dataset.y_train_l,
                'ohe': dataset.y_train_ohe,
                'raw': dataset.y_train
                }
            
            y_dict_test = {
                'label': dataset.y_test_l,
                'ohe': dataset.y_test_ohe,
                'raw': dataset.y_test
            }
            
            y_dict_holdout = {
                'label': dataset.y_holdout_l,
                'ohe': dataset.y_holdout_ohe,
                'raw': dataset.y_holdout
            }
            
            y_train = y_dict_train[self.y_format]
            y_test = y_dict_test[self.y_format]
            y_holdout = y_dict_holdout[self.y_format]
            
            if self.nn == True:
                
                model = self.pipeline.fit(dataset.X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data=(dataset.X_train_scaled, y_train))
                
                y_train_preds = self.pipeline.predict(dataset.X_train_scaled)
                y_test_preds = self.pipeline.predict(dataset.X_test_scaled)
                y_holdout_preds = self.pipeline.predict(dataset.X_holdout_scaled)
                
                hold_val = self.pipeline.evaluate(dataset.X_holdout_scaled, y_holdout)
            
            else:
                model = self.pipeline.fit(dataset.X_train, y_train)
                
                y_train_preds = model.predict(dataset.X_train)
                y_test_preds = model.predict(dataset.X_test)
                y_holdout_preds = model.predict(dataset.X_holdout)
            
            if self.nn==False:
                
                self.data_dict[dataset.name] = {
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_holdout': y_holdout,
                    'y_train_preds': y_train_preds, 
                    'y_test_preds': y_test_preds,
                    'y_holdout_preds': y_holdout_preds,   
                }
            
            else:
                self.data_dict[dataset.name] = {
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_holdout': y_holdout,
                    'y_train_preds': y_train_preds, 
                    'y_test_preds': y_test_preds,
                    'y_holdout_preds': y_holdout_preds,
                    'model_history': model.history,
                    'holdout_history': hold_val
                }
        
        if self.nn == False:
            self.calc_scores(data=set_list)
        else:
            self.calc_nn_scores(data=set_list)
        self.process_time = time.time() - start_time
        print("--- %s seconds to process ---" % (self.process_time))
    
    def calc_scores(self, data=None, data_type='test'):
        
        '''
        Calculates all major scores on each dataset for comparison for non neural network datasets. Will return an error if the model is a neural network.
        
        Parameters
        ----------
        
        data : List
            A list of processed TextSet objects. If None is passed, then the model will train on every model in the ModelComparison object
        data_type : String - 'test' or 'holdout'
            Determines the values compared in the final score comparison. If 'test', then the test scores will be compared. If 'holdout', then holdout scores will be shown.
        
        Returns
        -------
        
        all_scores : DataFrame
            All train and test scores for all datasets in a single dataframe showing scores for Accuracy, Precision, Recall and F1 Score. All values are calculated using the Macro method.
        score_comparison : DataFrame
            Test scores for each dataset compared in a single dataframe with the top value in each row highlighted.
        '''
        
        df_list = []
        
        title = data_type.title()
        
        if data == None:
            set_list = self.data_list
        
        else:
            set_list = data
        
        for dataset in set_list:
            
            y_train = self.data_dict[dataset.name]['y_train']
            y_train_preds = self.data_dict[dataset.name]['y_train_preds']
        
            if data_type == 'test':
                
                y_val = self.data_dict[dataset.name]['y_test']
                y_val_preds = self.data_dict[dataset.name]['y_test_preds']
            
            elif data_type == 'holdout':
            
                y_val = self.data_dict[dataset.name]['y_holdout']
                y_val_preds = self.data_dict[dataset.name]['y_holdout_preds']
            
            dictionary = {
                'Accuracy': [accuracy_score(y_train, y_train_preds), accuracy_score(y_val, y_val_preds)],
                'Precision (Macro)': [precision_score(y_train, y_train_preds, average='macro'), precision_score(y_val, y_val_preds, average='macro')],
                'Recall (Macro)': [recall_score(y_train, y_train_preds, average='macro'), recall_score(y_val, y_val_preds, average='macro')],
                'F1 (Macro)': [f1_score(y_train, y_train_preds, average='macro'), f1_score(y_val, y_val_preds, average='macro')],
            }
            
            df = pd.DataFrame.from_dict(dictionary, orient='index', columns=[f'{dataset.name} Train',f'{dataset.name} {title}'])
            
            self.data_dict[dataset.name][f'{data_type} scores'] = df
            df_list.append(df)
        
        self.all_scores = pd.concat(df_list, axis=1)
        
        self.score_comparison = self.all_scores[[x for x in self.all_scores.columns if x.endswith(title)]]
        
        self.score_comparison = self.score_comparison.style.highlight_max(color='lightgreen', axis=1)
        
    def calc_nn_scores(self, data=None):
        
        '''
        Calculates accuracy and loss scores for each dataset if the model is a neural network. Will produce an error if the model is not a neural network.
        
        Parameters
        ----------
        
        data : List
            A list of processed TextSet objects. If None is passed, then the model will train on every model in the ModelComparison object
        
        Returns
        -------
        
        score_comparison : DataFrame
            A dataframe comparing the accuracy and loss scores on the holdout data for each dataset
        '''

        df_list = []
        
        if data == None:
            set_list = self.data_list
        
        else:
            set_list = data

        for dataset in set_list:

            hist = self.data_dict[dataset.name]['holdout_history']

            dictionary = {
                'Loss': hist[0],
                'Accuracy': hist[1]
            }

            df = pd.DataFrame.from_dict(dictionary, orient='index', columns=[f'{dataset.name} Holdout'])

            df_list.append(df)

        self.score_comparison = pd.concat(df_list, axis=1)
            
    def compare_confusion(self, data_type='test', data=None):
        
        '''
        Plots a confusion matrix for either the Test or Holdout data for each dataset. Bear in mind, it will fit the model with all of the datsets again.
        
        Parameters
        ----------
        
        data_type : String - 'test' or 'holdout'
            Determines which dataset will be used in the confusion matrix
        data : List
            A list of processed TextSet objects. If None is passed, then the model will train on every model in the ModelComparison object
            
        Returns
        -------
        
        A confusion matrix for each dataset passed in data
        '''
        
        title = data_type.title()
        
        if data == None:
            set_list = self.data_list
        
        else:
            set_list = data
        
        for dataset in set_list:
            
            X_train = dataset.X_train
            y_train = self.data_dict[dataset.name]['y_train']
            
            self.pipeline.fit(X_train, y_train)
            
            if data_type == 'train':
                X_val = dataset.X_train
                y_val = self.data_dict[dataset.name]['y_train']
            
            elif data_type == 'test':
                X_val = dataset.X_test
                y_val = self.data_dict[dataset.name]['y_test']
                
            elif data_type == 'holdout':
                X_val = dataset.X_holdout
                y_val = self.data_dict[dataset.name]['y_holdout']
        
            plot_confusion_matrix(self.pipeline, X_val, y_val, cmap=plt.cm.Blues)
            plt.title(f'{dataset.name} {title}')
            plt.show()
        
    def plot_roc_curve(self, data_type='test', include_train=True, data=None):
        
        '''
        Plots the ROC curve for the results of the model. If a multi-class model it will return an error
        
        Parameters
        ----------
        
        data_type : String - 'test' or 'holdout'
            Determines which dataset will be used in the ROC Plot
        include_train : Boolean
            If True is passed, the ROC plot will include the training data and the validation data
        data : List
            A list of processed TextSet objects. If None is passed, then all datasets will be included in the final plot.
        
        Returns
        -------
        
        A plot of the ROC curve of the train and validation data
        '''
        
        plt.figure(figsize=(7,7))
        ax = plt.gca()
        
        if data == None:
            set_list = self.data_list
        else:
            set_list = [self.ref_dict[name] for name in data]
        
        for dataset in set_list:
            if data_type == 'test':
                y_val = self.data_dict[dataset.name]['y_test']
                y_val_preds = self.data_dict[dataset.name]['y_test_preds']
            elif data_type == 'holdout':
                y_val = self.data_dict[dataset.name]['y_holdout']
                y_val_preds = self.data_dict[dataset.name]['y_holdout_preds']
            
            fpr, tpr, threshold = roc_curve(y_val, y_val_preds)
            plt.plot(fpr, tpr, label=f'{dataset.name} {data_type}', ax=ax)
            
            if include_train == True:
                y_train = self.data_dict[dataset.name]['y_train']
                y_train_preds = self.data_dict[dataset.name]['y_train_preds']
                
                fpr, tpr, threshold = roc_curve(y_train, y_train_preds)
                plt.plot(fpr, tpr, label=f'{dataset.name} train', ax=ax)
                
                
        plt.title(f'ROC Curve for {data_type} data')
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        
    def plot_val_history(self, data=None):
        
        '''
        Plots the accuracy and loss curves for both the training and the testing data for a neural network model.
        
        Parameters
        ----------
        
        data : List
            A list of processed TextSet objects. If None is passed, then all datasets will be included in the final plots.
        
        Returns
        -------
        
        A subplot of the training and validation loss and accuracy values from the fit Neural Network model
        
        '''
        
        if data == None:
            set_list = self.data_list
        else:
            set_list = [self.ref_dict[name] for name in data]
        
        for dataset in set_list:
            history = self.data_dict[dataset.name]['model_history']
            fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(12, 4))
            ax1.plot(history['val_loss'])
            ax1.plot(history['loss'])
            ax1.legend(['val_loss', 'loss'])
            ax1.set_title(f'{dataset.name} Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')

            ax2.plot(history['val_acc'])
            ax2.plot(history['acc'])
            ax2.legend(['val_acc', 'acc'])
            ax2.set_title(f'{dataset.name} Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            plt.show()
        
        
    