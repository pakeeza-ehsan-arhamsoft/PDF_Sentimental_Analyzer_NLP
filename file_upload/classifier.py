import re
import string
import warnings

import PyPDF2
import nltk
import numpy as np
import pandas as pd
import text2emotion
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from textblob import classifiers

warnings.filterwarnings('ignore')


class DataProcessing:
    def __init__(self):
        self.docs = []
        # Tokenization of text
        self.tokenizer = ToktokTokenizer()
        # Setting English stopwords
        self.stopword_list = nltk.corpus.stopwords.words('english')
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()

    def tokenization(self, doc: list) -> list:
        for i in range(0, len(doc)):
            text = self.strip_html(doc[i])
            text = self.remove_special_characters(doc[i])
            text = self.remove_between_square_brackets(doc[i])
            sentences = sent_tokenize(str(text))
            doc[i] = sentences

        return doc

    def strip_html(self, text: string) -> string:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    # Removing the square brackets
    def remove_between_square_brackets(self, text: string) -> string:
        return re.sub('\[[^]]*\]', '', text)

    def remove_special_characters(self, text: string) -> string:
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern, '', str(text))
        return text

    # Stemming the text
    def simple_stemmer(self, text: string) -> string:
        ps = nltk.porter.PorterStemmer()
        text2 = [ps.stem(word) for word in text.split()]
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    # removing the stopwords
    def remove_stopwords(self, text: string, is_lower_case: bool = False) -> string:
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:

            filtered_tokens = [token for token in tokens if token not in self.stopword_list and len(token) > 2]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in self.stopword_list and len(token) > 2]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def text_normalize(self, text: string) -> string:
        term_word = []
        # spell checker
        norm_spelling = TextBlob(text)
        term_word.append(str(norm_spelling.correct()))
        term_word = ' '.join(term_word)
        return term_word

    def lemmatizer(self, text: string) -> string:
        term_word2 = [self.wordnet_lemmatizer.lemmatize(d, pos='v') for d in text.split()]
        term_word2 = ' '.join(term_word2)
        return term_word2

    # Removing the noisy text
    def denoise_text(self, text: string) -> string:
        #         text = self.strip_html(text)
        #         text = self.remove_between_square_brackets(text)
        #         text = self.remove_special_characters(text)
        text = self.simple_stemmer(text)
        text = self.remove_stopwords(text)
        text = self.lemmatizer(text)

        text = self.text_normalize(text)
        return text


class ToneDetector(DataProcessing):
    def __init__(self):
        super().__init__()

    def processing(self, docs: list) -> list:
        doc = self.tokenization(docs)
        return doc

    def data_cleaning(self, df: object) -> object:
        # Apply function on review column
        for i in range(0, len(df['pdf_data'])):
            data = self.denoise_text(str(df['pdf_data'][i]))
            df['pdf_data'][i] = data
        return df

    def check_Tone(self, df: object) -> object:
        senti = []
        for i in range(0, len(df['pdf_data'])):
            blob = TextBlob(str(df['pdf_data'][i]))
            score = blob.sentiment[0]
            if score < 0:
                senti.append("Negative")
            elif score == 0:
                senti.append("Neutral")
            else:
                senti.append("Positive")

        df = pd.DataFrame(list(zip(df['pdf_data'], senti)), columns=['pdf_data', 'Sentimental'])
        return df

    def check_Emotion(self, df: object) -> object:
        emotions = []
        for i in range(0, len(df['pdf_data'])):
            emotions.append(text2emotion.get_emotion(str(df['pdf_data'][i])))

        df = pd.DataFrame(list(zip(df['pdf_data'], df['Sentimental'], emotions)),
                          columns=['pdf_data', 'Sentimental', 'Emotions'])
        return df

    def classify(self, df: object) -> object:
        training = []
        for i in range(15, len(df['pdf_data'])):
            training.append((df['pdf_data'][i], df['Sentimental'][i]))
        testing = []
        for i in range(0, 15):
            testing.append((df['pdf_data'][i], df['Sentimental'][i]))
        classifier = classifiers.NaiveBayesClassifier(training)
        y_pred = classifier.accuracy(testing)
        print("\n Accuracy: ", y_pred)
        classifier.show_informative_features(3)


class PDFReader(ToneDetector):
    def __init__(self):
        super().__init__()
        self.doc = []
        self.pages = []

    def read_pdf(self, file_url) -> object:
        # creating a pdf file object
        file_url = "." + file_url
        print("url: ", file_url)
        pdfFileObj = open(file_url, 'rb')

        # creating a pdf reader object
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        # printing number of pages in pdf file
        pg = pdfReader.numPages
        txt = ""
        for j in range(0, pg):
            pageObj = pdfReader.getPage(j)
            txt += pageObj.extractText()
        txt = np.char.replace(txt, ',', '')
        self.doc.append(str(txt))
        # closing the pdf file object
        pdfFileObj.close()
        df = self.make_dataframe(self.doc)
        return df

    def make_dataframe(self, doc: list) -> object:
        doc = self.processing(doc)
        for i in doc:
            df = pd.DataFrame(i, columns=['pdf_data'])
        return df

    def CheckTone(self,file_url) -> object:
        df = self.read_pdf(file_url)
        df = self.check_Tone(df)
        df = self.check_Emotion(df)
        print("Sentimental Analyzer without Stemming and Lemmetizer: \n")
        print(df.head())
        self.classify(df)

        df = self.data_cleaning(df)
        df = self.check_Tone(df)
        nan_value = float("NaN")
        # Convert NaN values to empty string
        df.replace("", nan_value, inplace=True)
        df.dropna(subset=["pdf_data"], inplace=True)
        df = df.reset_index(drop=True)
        df = self.check_Emotion(df)
        print("\nSentimental Analyzer with Stemming and Lemmetizer: \n")
        print(df.head())
        # self.classify(df)
        return df
