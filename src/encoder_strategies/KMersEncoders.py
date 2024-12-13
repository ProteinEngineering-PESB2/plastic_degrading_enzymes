from encoder_strategies.Encoders import Encoders
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class KMersEncoders(Encoders):

    def __init__(
            self, 
            dataset=None, 
            sequence_column=None,
            size_kmer=3,
            ignore_columns=[]) -> None:
        
        super().__init__(
            dataset=dataset, 
            sequence_column=sequence_column, 
            ignore_columns=ignore_columns)
        
        self.size_kmer = size_kmer
    
    def kmer(self, seq,seq_length,kmer_length=3):
        kmer_words = [seq[i:i+seq_length] for i in range(len(seq)-seq_length+1)]
        return ' '.join(kmer_words)

    def process_dataset(self):

        tfidfvector = TfidfVectorizer()

        self.dataset['kmer_sequence'] = self.dataset[self.sequence_column].apply(
            lambda x: self.kmer(x,self.size_kmer))
        
        processed_data = tfidfvector.fit_transform(
            self.dataset['kmer_sequence']).astype('float32')
        
        self.coded_dataset = pd.DataFrame(data=processed_data.toarray(), 
                       columns=tfidfvector.get_feature_names_out())
        
        self.coded_dataset.columns = [column.upper() for column in self.coded_dataset.columns]
        self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values

        for column in self.ignore_columns:
            self.coded_dataset[column] = self.dataset[column].values
        

    