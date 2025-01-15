from lib.Encoders import Encoders
import pandas as pd

class PhysicochemicalEncoder(Encoders):

    def __init__(
            self, 
            dataset=None, 
            sequence_column=None, 
            ignore_columns=[],
            max_length=1024,
            name_property="ANDN920101",
            df_properties=None) -> None:
        
        super().__init__(
            dataset=dataset, 
            sequence_column=sequence_column, 
            ignore_columns=ignore_columns,
            max_length=max_length)
        
        self.name_property = name_property
        self.df_properties = df_properties
    
    def __encoding_residue(self, residue):

        return self.df_properties[self.name_property][residue]
        
    def __encoding_sequence(self, sequence):

        sequence = sequence.upper()
        sequence_encoding = []

        for i in range(len(sequence)):
            residue = sequence[i]
            response_encoding = self.__encoding_residue(residue)
            if response_encoding != False:
                sequence_encoding.append(response_encoding)

        for k in range(len(sequence_encoding), self.max_length+1):
            sequence_encoding.append(0)

        return sequence_encoding
    
    def __encoding_dataset(self):

        print("Encoding and Processing results")

        matrix_data = []
        for index in self.dataset.index:
            sequence_encoder = self.__encoding_sequence(sequence=self.dataset[self.sequence_column][index])
            matrix_data.append(sequence_encoder)

        print("Creating dataset")
        header = ['p_{}'.format(i) for i in range(len(matrix_data[0]))]
        print("Export dataset")

        self.df_data_encoded = pd.DataFrame(matrix_data, columns=header)

        for column in self.ignore_columns:
            self.df_data_encoded[column] = self.dataset[column].values
    
    def run_process(self):

        if self.status:
            self.__encoding_dataset()