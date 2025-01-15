from lib.Constant import * 
import pandas as pd

class Encoders(object):

    def __init__(
            self, 
            dataset=None,
            sequence_column=None,
            ignore_columns=None,
            max_length=1024) -> None:
        
        self.dataset = dataset
        self.sequence_column = sequence_column
        self.ignore_columns = ignore_columns
        self.max_length = max_length

        self.status = True
        self.message = ""
                
        self.coded_dataset = pd.DataFrame()
        for column in self.ignore_columns:
            self.coded_dataset[column] = self.dataset[column].values

        self.make_revisions()
        
    def make_revisions(self):

        if self.sequence_column not in self.dataset.columns:
            self.status = False
            self.message = "Non sequence column identified on dataset columns"
        else:
            self.check_canonical_residues()
            self.process_length_sequences()

    def check_canonical_residues(self):
        
        print("Checking canonical residues in the dataset")
        canon_sequences = []

        for index in self.dataset.index:
            is_canon=True

            sequence = self.dataset[self.sequence_column][index]

            for residue in sequence:
                if residue not in LIST_RESIDUES:
                    is_canon = False
                    break
            
            canon_sequences.append(is_canon)
        
        self.dataset["is_canon"] = canon_sequences
        self.dataset = self.dataset[self.dataset["is_canon"]]
    
    def process_length_sequences(self):
        print("Estimating lenght in protein sequences")
        self.dataset["length_sequence"] = self.dataset[self.sequence_column].str.len()
        
        print("Evaluating length in protein sequences")
        self.dataset["is_valid_length"] = (self.dataset["length_sequence"]<=self.max_length).astype(int).values
        self.dataset = self.dataset[self.dataset["is_valid_length"]==1]