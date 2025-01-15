from modlamp.descriptors import GlobalDescriptor

def get_mw(sequence):
    """Molecular Weight"""
    try:
        desc = GlobalDescriptor([sequence])
        desc.calculate_MW(amide=True)
        return round(desc.descriptor[0][0], 4)
    except:
        return None

def get_isoelectric_point(sequence):
    """Isoelectric point"""
    try:
        desc = GlobalDescriptor([sequence])
        desc.isoelectric_point(amide=True)
        return round(desc.descriptor[0][0], 4)
    except:
        return None

def get_charge_density(sequence):
    """Charge density"""
    try:
        desc = GlobalDescriptor([sequence])
        desc.charge_density(ph=7, amide=True)
        return round(desc.descriptor[0][0], 5)
    except:
        return None

def get_charge(sequence):
    """Charge"""
    try:
        desc = GlobalDescriptor([sequence])
        desc.calculate_charge(ph=7, amide=True)
        return round(desc.descriptor[0][0], 4)
    except:
        return None

def get_instability_index(sequence):
    """Instability index"""
    try:
        desc = GlobalDescriptor([sequence])
        desc.instability_index()
        return round(desc.descriptor[0][0], 4)
    except:
        return None

def get_aromaticity(sequence):
    """Aromaticity"""
    try:
        desc = GlobalDescriptor([sequence])
        desc.aromaticity()
        return round(desc.descriptor[0][0], 4)
    except:
        return None

def get_aliphatic_index(sequence):
    """Aliphatic index"""
    try:
        desc = GlobalDescriptor([sequence])
        desc.aliphatic_index()
        return round(desc.descriptor[0][0], 4)
    except:
        return None

def get_boman_index(sequence):
    """Boman index"""
    try:
        desc = GlobalDescriptor([sequence])
        desc.boman_index()
        return round(desc.descriptor[0][0], 4)
    except:
        return None

def get_hydrophobic_ratio(sequence):
    """Hydrophobic ratio"""
    try:
        desc = GlobalDescriptor([sequence])
        desc.hydrophobic_ratio()
        return round(desc.descriptor[0][0], 4)
    except:
        return None

def get_frequency_residue(sequence, residue):
    return round(sequence.count(residue)/len(sequence), ndigits=4)