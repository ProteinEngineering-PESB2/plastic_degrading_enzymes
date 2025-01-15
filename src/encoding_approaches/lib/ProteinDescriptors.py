from lib.CalculatePeptideProperties import *
from lib.Constant import *

class ProteinDescriptors(object):

    def __init__(
            self,
            dataset=None,
            column_seq=None) -> None:
        
        self.dataset = dataset
        self.column_seq = column_seq

        self.status = True
        self.message = ""

    def apply_physicochemical_properties(self):

        self.dataset["Molecular Weight"] = self.dataset[self.column_seq].apply(
            lambda x: get_mw(x)
        )

        self.dataset["Isoelectric point"] = self.dataset[self.column_seq].apply(
            lambda x: get_isoelectric_point(x)
        )

        self.dataset["Charge density"] = self.dataset[self.column_seq].apply(
            lambda x: get_charge_density(x)
        )

        self.dataset["Charge"] = self.dataset[self.column_seq].apply(
            lambda x: get_charge(x)
        )

        self.dataset["Instability index"] = self.dataset[self.column_seq].apply(
            lambda x: get_instability_index(x)
        )

        self.dataset["Aromaticity"] = self.dataset[self.column_seq].apply(
            lambda x: get_aromaticity(x)
        )

        self.dataset["Aliphatic index"] = self.dataset[self.column_seq].apply(
            lambda x: get_aliphatic_index(x)
        )

        self.dataset["Boman index"] = self.dataset[self.column_seq].apply(
            lambda x: get_boman_index(x)
        )

        self.dataset["Hydrophobic ratio"] = self.dataset[self.column_seq].apply(
            lambda x: get_hydrophobic_ratio(x)
        )

        for residue in LIST_RESIDUES:

            name_col = f"freq_{residue}"

            self.dataset[name_col] = self.dataset[self.column_seq].apply(
                lambda x : get_frequency_residue(x, residue)
            )

    