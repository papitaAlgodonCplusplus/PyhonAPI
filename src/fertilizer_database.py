#!/usr/bin/env python3
"""
UPDATED COMPLETE FERTILIZER DATABASE MODULE WITH MICRONUTRIENT SUPPORT
Comprehensive fertilizer composition database with intelligent matching and complete micronutrient coverage
"""

from typing import Dict, Optional, List, Any
from models import Fertilizer, FertilizerComposition, FertilizerChemistry

class EnhancedFertilizerDatabase:
    """Complete fertilizer composition database with intelligent pattern matching and micronutrient support"""
    
    def __init__(self):
        self.fertilizer_data = {
            # ACIDS
            'acido_nitrico': {
                'patterns': ['acido nitrico', 'nitric acid', 'hno3', 'acido nítrico'],
                'formula_patterns': ['HNO3'],
                'composition': {
                    'formula': 'HNO3',
                    'mw': 63.01,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 22.23, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'acido_fosforico': {
                'patterns': ['acido fosforico', 'acido fosfórico', 'phosphoric acid', 'h3po4'],
                'formula_patterns': ['H3PO4'],
                'composition': {
                    'formula': 'H3PO4',
                    'mw': 97.99,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 31.61, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'acido_sulfurico': {
                'patterns': ['acido sulfurico', 'acido sulfúrico', 'sulfuric acid', 'h2so4'],
                'formula_patterns': ['H2SO4'],
                'composition': {
                    'formula': 'H2SO4',
                    'mw': 98.08,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 32.69, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            
            # NITRATES
            'nitrato_calcio': {
                'patterns': ['nitrato de calcio', 'calcium nitrate', 'nitrato calcio'],
                'formula_patterns': ['CA(NO3)2', 'CA(NO3)2.4H2O', 'CA(NO3)2.2H2O', 'Ca(NO3)2'],
                'composition': {
                    'formula': 'Ca(NO3)2.4H2O',
                    'mw': 236.15,
                    'cations': {'Ca': 16.97, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 11.86, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'nitrato_potasio': {
                'patterns': ['nitrato de potasio', 'potassium nitrate', 'nitrato potasio'],
                'formula_patterns': ['KNO3'],
                'composition': {
                    'formula': 'KNO3',
                    'mw': 101.1,
                    'cations': {'Ca': 0, 'K': 38.67, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 13.85, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'nitrato_amonio': {
                'patterns': ['nitrato de amonio', 'ammonium nitrate', 'nitrato amonio'],
                'formula_patterns': ['NH4NO3'],
                'composition': {
                    'formula': 'NH4NO3',
                    'mw': 80.04,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 22.5, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 35.0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'nitrato_magnesio': {
                'patterns': ['nitrato de magnesio', 'magnesium nitrate', 'nitrato magnesio'],
                'formula_patterns': ['MG(NO3)2', 'MG(NO3)2.6H2O', 'Mg(NO3)2'],
                'composition': {
                    'formula': 'Mg(NO3)2.6H2O',
                    'mw': 256.41,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 9.48, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 10.93, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            
            # SULFATES
            'sulfato_amonio': {
                'patterns': ['sulfato de amonio', 'ammonium sulfate', 'sulfato amonio'],
                'formula_patterns': ['(NH4)2SO4', 'NH4)2SO4'],
                'composition': {
                    'formula': '(NH4)2SO4',
                    'mw': 132.14,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 27.28, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 21.21, 'S': 24.26, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'sulfato_potasio': {
                'patterns': ['sulfato de potasio', 'potassium sulfate', 'sulfato potasio'],
                'formula_patterns': ['K2SO4'],
                'composition': {
                    'formula': 'K2SO4',
                    'mw': 174.26,
                    'cations': {'Ca': 0, 'K': 44.87, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 18.39, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'sulfato_magnesio': {
                'patterns': ['sulfato de magnesio', 'magnesium sulfate', 'sulfato magnesio', 'sal de epsom', 'epsom salt'],
                'formula_patterns': ['MGSO4', 'MGSO4.7H2O', 'MgSO4', 'MgSO4.7H2O'],
                'composition': {
                    'formula': 'MgSO4.7H2O',
                    'mw': 246.47,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 9.87, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 13.01, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'sulfato_calcio': {
                'patterns': ['sulfato de calcio', 'calcium sulfate', 'sulfato calcio', 'yeso'],
                'formula_patterns': ['CASO4', 'CASO4.2H2O', 'CaSO4', 'CaSO4.2H2O'],
                'composition': {
                    'formula': 'CaSO4.2H2O',
                    'mw': 172.17,
                    'cations': {'Ca': 23.28, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 18.62, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            
            # PHOSPHATES
            'fosfato_monopotasico': {
                'patterns': ['fosfato monopotasico', 'fosfato monopotásico', 'monopotassium phosphate', 'kh2po4', 'mkp'],
                'formula_patterns': ['KH2PO4'],
                'composition': {
                    'formula': 'KH2PO4',
                    'mw': 136.09,
                    'cations': {'Ca': 0, 'K': 28.73, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 22.76, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'fosfato_dipotasico': {
                'patterns': ['fosfato dipotasico', 'fosfato dipotásico', 'dipotassium phosphate', 'k2hpo4', 'dkp'],
                'formula_patterns': ['K2HPO4'],
                'composition': {
                    'formula': 'K2HPO4',
                    'mw': 174.18,
                    'cations': {'Ca': 0, 'K': 44.93, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 17.79, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'fosfato_monoamonico': {
                'patterns': ['fosfato monoamonico', 'fosfato monoamónico', 'monoammonium phosphate', 'map'],
                'formula_patterns': ['NH4H2PO4'],
                'composition': {
                    'formula': 'NH4H2PO4',
                    'mw': 115.03,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 15.65, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 12.18, 'S': 0, 'Cl': 0, 'P': 26.93, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'fosfato_diamonico': {
                'patterns': ['fosfato diamonico', 'fosfato diamónico', 'diammonium phosphate', 'dap'],
                'formula_patterns': ['(NH4)2HPO4'],
                'composition': {
                    'formula': '(NH4)2HPO4',
                    'mw': 132.06,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 27.28, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 21.22, 'S': 0, 'Cl': 0, 'P': 23.47, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            
            # CHLORIDES
            'cloruro_calcio': {
                'patterns': ['cloruro de calcio', 'calcium chloride', 'cloruro calcio'],
                'formula_patterns': ['CACL2', 'CACL2.2H2O', 'CaCl2', 'CaCl2.2H2O'],
                'composition': {
                    'formula': 'CaCl2.2H2O',
                    'mw': 147.01,
                    'cations': {'Ca': 27.26, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 48.23, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'cloruro_potasio': {
                'patterns': ['cloruro de potasio', 'potassium chloride', 'cloruro potasio', 'muriato de potasio'],
                'formula_patterns': ['KCL', 'KCl'],
                'composition': {
                    'formula': 'KCl',
                    'mw': 74.55,
                    'cations': {'Ca': 0, 'K': 52.44, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 47.56, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'cloruro_magnesio': {
                'patterns': ['cloruro de magnesio', 'magnesium chloride', 'cloruro magnesio'],
                'formula_patterns': ['MGCL2', 'MGCL2.6H2O', 'MgCl2', 'MgCl2.6H2O'],
                'composition': {
                    'formula': 'MgCl2.6H2O',
                    'mw': 203.30,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 11.96, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 34.87, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },

            # ===== IRON SOURCES (Complete Coverage) =====
            'quelato_hierro': {
                'patterns': ['quelato de hierro', 'iron chelate', 'fe-edta', 'fe edta', 'feeedta', 'iron edta', 'chelato hierro'],
                'formula_patterns': ['FE-EDTA', 'C10H12FeN2NaO8'],
                'composition': {
                    'formula': 'C10H12FeN2NaO8',
                    'mw': 367.05,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 6.27, 'NH4': 0, 'Fe': 13.0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 7.63, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'sulfato_hierro': {
                'patterns': ['sulfato de hierro', 'iron sulfate', 'sulfato ferroso', 'feso4', 'ferrous sulfate', 'hierro sulfato'],
                'formula_patterns': ['FESO4', 'FESO4.7H2O', 'FeSO4', 'FeSO4.7H2O'],
                'composition': {
                    'formula': 'FeSO4.7H2O',
                    'mw': 278.01,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 20.09, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 11.53, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'cloruro_hierro': {
                'patterns': ['cloruro de hierro', 'iron chloride', 'ferric chloride', 'fecl3', 'hierro cloruro'],
                'formula_patterns': ['FECL3', 'FeCl3.6H2O', 'FeCl3'],
                'composition': {
                    'formula': 'FeCl3.6H2O',
                    'mw': 270.30,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 20.66, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 39.35, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'quelato_hierro_dtpa': {
                'patterns': ['fe-dtpa', 'iron dtpa', 'quelato hierro dtpa', 'dtpa iron'],
                'formula_patterns': ['FE-DTPA'],
                'composition': {
                    'formula': 'Fe-DTPA',
                    'mw': 447.16,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 5.15, 'NH4': 0, 'Fe': 12.5, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 6.26, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },

            # ===== MANGANESE SOURCES (Complete Coverage) =====
            'sulfato_manganeso': {
                'patterns': ['sulfato de manganeso', 'manganese sulfate', 'sulfato manganeso', 'mnso4', 'manganeso sulfato'],
                'formula_patterns': ['MNSO4', 'MNSO4.4H2O', 'MnSO4', 'MnSO4.4H2O', 'MNSO4.H2O'],
                'composition': {
                    'formula': 'MnSO4.4H2O',
                    'mw': 223.06,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 24.63, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 14.37, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'cloruro_manganeso': {
                'patterns': ['cloruro de manganeso', 'manganese chloride', 'mncl2', 'manganeso cloruro'],
                'formula_patterns': ['MNCL2', 'MnCl2.4H2O', 'MnCl2'],
                'composition': {
                    'formula': 'MnCl2.4H2O',
                    'mw': 197.91,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 27.76, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 35.84, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'quelato_manganeso': {
                'patterns': ['quelato de manganeso', 'manganese chelate', 'mn-edta', 'mn edta', 'manganeso quelato'],
                'formula_patterns': ['MN-EDTA'],
                'composition': {
                    'formula': 'MnEDTA',
                    'mw': 345.08,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 6.67, 'NH4': 0, 'Fe': 0, 'Mn': 15.92, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 8.12, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },

            # ===== ZINC SOURCES (Complete Coverage) =====
            'sulfato_zinc': {
                'patterns': ['sulfato de zinc', 'zinc sulfate', 'sulfato zinc', 'znso4', 'zinc sulfato'],
                'formula_patterns': ['ZNSO4', 'ZNSO4.7H2O', 'ZnSO4', 'ZnSO4.7H2O', 'ZNSO4.H2O'],
                'composition': {
                    'formula': 'ZnSO4.7H2O',
                    'mw': 287.56,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 22.74, 'Cu': 0},
                    'anions': {'N': 0, 'S': 11.15, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'cloruro_zinc': {
                'patterns': ['cloruro de zinc', 'zinc chloride', 'zncl2', 'zinc cloruro'],
                'formula_patterns': ['ZNCL2', 'ZnCl2'],
                'composition': {
                    'formula': 'ZnCl2',
                    'mw': 136.30,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 47.96, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 52.04, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'quelato_zinc': {
                'patterns': ['quelato de zinc', 'zinc chelate', 'zn-edta', 'zn edta', 'zinc quelato'],
                'formula_patterns': ['ZN-EDTA'],
                'composition': {
                    'formula': 'ZnEDTA',
                    'mw': 351.56,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 6.54, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 18.60, 'Cu': 0},
                    'anions': {'N': 7.97, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },

            # ===== COPPER SOURCES (Complete Coverage) =====
            'sulfato_cobre': {
                'patterns': ['sulfato de cobre', 'copper sulfate', 'sulfato cobre', 'cuso4', 'cobre sulfato'],
                'formula_patterns': ['CUSO4', 'CUSO4.5H2O', 'CuSO4', 'CuSO4.5H2O'],
                'composition': {
                    'formula': 'CuSO4.5H2O',
                    'mw': 249.69,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 25.45},
                    'anions': {'N': 0, 'S': 12.84, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'cloruro_cobre': {
                'patterns': ['cloruro de cobre', 'copper chloride', 'cucl2', 'cobre cloruro'],
                'formula_patterns': ['CUCL2', 'CuCl2.2H2O', 'CuCl2'],
                'composition': {
                    'formula': 'CuCl2.2H2O',
                    'mw': 170.48,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 37.27},
                    'anions': {'N': 0, 'S': 0, 'Cl': 41.61, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'quelato_cobre': {
                'patterns': ['quelato de cobre', 'copper chelate', 'cu-edta', 'cu edta', 'cobre quelato'],
                'formula_patterns': ['CU-EDTA'],
                'composition': {
                    'formula': 'CuEDTA',
                    'mw': 347.76,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 6.62, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 18.28},
                    'anions': {'N': 8.06, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },

           # ===== BORON SOURCES (Complete Coverage) =====
            'acido_borico': {
                'patterns': ['acido borico', 'ácido bórico', 'boric acid', 'h3bo3', 'boro acido'],
                'formula_patterns': ['H3BO3'],
                'composition': {
                    'formula': 'H3BO3',
                    'mw': 61.83,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 17.48, 'Mo': 0}
                }
            },
            'borax': {
                'patterns': ['borax', 'sodium borate', 'borato de sodio', 'na2b4o7', 'tetraborato de sodio'],
                'formula_patterns': ['NA2B4O7', 'Na2B4O7.10H2O'],
                'composition': {
                    'formula': 'Na2B4O7.10H2O',
                    'mw': 381.37,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 12.06, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 11.34, 'Mo': 0}
                }
            },
            'soluboro': {
                'patterns': ['soluboro', 'solubor', 'etanolamina borato', 'ethanolamine borate'],
                'formula_patterns': ['C2H8BNO3'],
                'composition': {
                    'formula': 'C2H8BNO3',
                    'mw': 104.90,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 13.35, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 10.3, 'Mo': 0}
                }
            },

            # ===== MOLYBDENUM SOURCES (Complete Coverage) =====
            'molibdato_sodio': {
                'patterns': ['molibdato de sodio', 'sodium molybdate', 'molibdato sodio', 'na2moo4', 'molibdeno sodio'],
                'formula_patterns': ['NA2MOO4', 'NA2MOO4.2H2O', 'Na2MoO4', 'Na2MoO4.2H2O'],
                'composition': {
                    'formula': 'Na2MoO4.2H2O',
                    'mw': 241.95,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 19.01, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 39.66}
                }
            },
            'molibdato_amonio': {
                'patterns': ['molibdato de amonio', 'ammonium molybdate', '(nh4)6mo7o24', 'molibdeno amonio'],
                'formula_patterns': ['(NH4)6MO7O24', '(NH4)6Mo7O24.4H2O'],
                'composition': {
                    'formula': '(NH4)6Mo7O24.4H2O',
                    'mw': 1235.86,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 8.78, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 6.82, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 54.34}
                }
            },
            'molibdato_calcio': {
                'patterns': ['molibdato de calcio', 'calcium molybdate', 'camoo4', 'molibdeno calcio'],
                'formula_patterns': ['CAMOO4', 'CaMoO4'],
                'composition': {
                    'formula': 'CaMoO4',
                    'mw': 200.02,
                    'cations': {'Ca': 20.04, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 47.96}
                }
            },

            # ===== MICRONUTRIENT MIXES (Professional Blends) =====
            'mix_micronutrientes': {
                'patterns': ['mix micronutrientes', 'micronutrient mix', 'cocktail micronutrientes', 'mezcla micronutrientes', 'micro mix'],
                'formula_patterns': ['MICRO-MIX'],
                'composition': {
                    'formula': 'Micronutrient Mix',
                    'mw': 500.0,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 8.0, 'NH4': 0, 'Fe': 7.0, 'Mn': 2.0, 'Zn': 1.5, 'Cu': 0.8},
                    'anions': {'N': 5.0, 'S': 2.0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 1.1, 'Mo': 0.15}
                }
            },
            'tenso_cocktail': {
                'patterns': ['tenso cocktail', 'tenso micro', 'professional micronutrient blend'],
                'formula_patterns': ['TENSO-MIX'],
                'composition': {
                    'formula': 'Professional Micro Blend',
                    'mw': 450.0,
                    'cations': {'Ca': 1.0, 'K': 2.0, 'Mg': 1.5, 'Na': 5.0, 'NH4': 0, 'Fe': 6.0, 'Mn': 1.8, 'Zn': 1.2, 'Cu': 0.6},
                    'anions': {'N': 3.0, 'S': 1.5, 'Cl': 0, 'P': 0.5, 'HCO3': 0, 'B': 0.9, 'Mo': 0.12}
                }
            },

            # ===== SPECIALIZED FERTILIZERS WITH MICRONUTRIENTS =====
            'nitrato_calcio_boro': {
                'patterns': ['nitrato de calcio con boro', 'calcium nitrate boron', 'nitrato calcio boro', 'calcium nitrate + b'],
                'formula_patterns': ['CA(NO3)2+B'],
                'composition': {
                    'formula': 'Ca(NO3)2.4H2O + B',
                    'mw': 236.15,
                    'cations': {'Ca': 16.5, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 11.5, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0.3, 'Mo': 0}
                }
            },
            'sulfato_magnesio_micro': {
                'patterns': ['sulfato magnesio micro', 'magnesium sulfate micronutrients', 'epsom plus micro'],
                'formula_patterns': ['MGSO4+MICRO'],
                'composition': {
                    'formula': 'MgSO4.7H2O + Micronutrients',
                    'mw': 246.47,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 9.5, 'Na': 1.0, 'NH4': 0, 'Fe': 0.5, 'Mn': 0.3, 'Zn': 0.2, 'Cu': 0.1},
                    'anions': {'N': 0, 'S': 12.8, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0.15, 'Mo': 0.02}
                }
            },

            # ===== CHELATED MICRONUTRIENT BLENDS =====
            'quelatos_mezclados': {
                'patterns': ['quelatos mezclados', 'mixed chelates', 'chelated micronutrient blend', 'edta mix'],
                'formula_patterns': ['EDTA-MIX'],
                'composition': {
                    'formula': 'Mixed EDTA Chelates',
                    'mw': 400.0,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 15.0, 'NH4': 0, 'Fe': 6.0, 'Mn': 3.0, 'Zn': 2.0, 'Cu': 1.0},
                    'anions': {'N': 20.0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            }
        }

    def find_fertilizer_composition(self, name: str, formula: str = "") -> Optional[Dict[str, Any]]:
        """
        Find fertilizer composition by intelligent name and formula matching including micronutrients
        """
        name_lower = name.lower().strip()
        formula_upper = formula.upper().strip()

        print(f"    🔍 Searching enhanced database for: name='{name_lower}', formula='{formula_upper}'")

        # Strategy 1: Try exact name pattern matching first
        for fert_key, fert_data in self.fertilizer_data.items():
            for pattern in fert_data['patterns']:
                if pattern in name_lower or name_lower in pattern:
                    print(f"    [FOUND] by name pattern: '{pattern}' -> {fert_data['composition']['formula']}")
                    return fert_data['composition']

        # Strategy 2: Try formula pattern matching
        if formula_upper:
            for fert_key, fert_data in self.fertilizer_data.items():
                for formula_pattern in fert_data['formula_patterns']:
                    if formula_pattern in formula_upper or formula_upper in formula_pattern:
                        print(f"    [FOUND] by formula pattern: '{formula_pattern}' -> {fert_data['composition']['formula']}")
                        return fert_data['composition']

        # Strategy 3: Enhanced fuzzy matching including ALL micronutrients
        enhanced_fuzzy_matches = {
            # Existing macronutrient matches
            'calcium': 'nitrato_calcio',
            'potassium': 'nitrato_potasio',
            'nitrate': 'nitrato_calcio',
            'phosphate': 'fosfato_monopotasico',
            'sulfate': 'sulfato_magnesio',
            'magnesium': 'sulfato_magnesio',
            
            # ENHANCED: Complete micronutrient matches
            'iron': 'quelato_hierro',
            'hierro': 'quelato_hierro',
            'fe': 'quelato_hierro',
            'ferrous': 'sulfato_hierro',
            'ferric': 'cloruro_hierro',
            
            'manganese': 'sulfato_manganeso',
            'manganeso': 'sulfato_manganeso',
            'mn': 'sulfato_manganeso',
            
            'zinc': 'sulfato_zinc',
            'zn': 'sulfato_zinc',
            
            'copper': 'sulfato_cobre',
            'cobre': 'sulfato_cobre',
            'cu': 'sulfato_cobre',
            
            'boron': 'acido_borico',
            'boro': 'acido_borico',
            'b': 'acido_borico',
            'boric': 'acido_borico',
            
            'molybdenum': 'molibdato_sodio',
            'molibdeno': 'molibdato_sodio',
            'mo': 'molibdato_sodio',
            'molybdate': 'molibdato_sodio',
            
            # Chelate patterns
            'chelate': 'quelato_hierro',
            'quelato': 'quelato_hierro',
            'edta': 'quelato_hierro',
            'dtpa': 'quelato_hierro_dtpa',
            
            # Mix patterns
            'micronutrient': 'mix_micronutrientes',
            'micronutrientes': 'mix_micronutrientes',
            'micro': 'mix_micronutrientes',
            'mix': 'mix_micronutrientes',
            'cocktail': 'mix_micronutrientes',
            'blend': 'mix_micronutrientes',
            'tenso': 'tenso_cocktail',
            
            # Specialized patterns
            'soluboro': 'soluboro',
            'solubor': 'soluboro',
            'borax': 'borax',
            'epsom': 'sulfato_magnesio'
        }
        
        for keyword, fert_key in enhanced_fuzzy_matches.items():
            if keyword in name_lower:
                if fert_key in self.fertilizer_data:
                    print(f"    [FOUND] by enhanced fuzzy matching: '{keyword}' -> {self.fertilizer_data[fert_key]['composition']['formula']}")
                    return self.fertilizer_data[fert_key]['composition']

        print(f"    [NOT FOUND] NO MATCH FOUND for '{name}' with formula '{formula}'")
        return None

    def create_fertilizer_from_database(self, name: str, formula: str = "") -> Optional[Fertilizer]:
        """
        Create a complete Fertilizer object from enhanced database
        """
        composition_data = self.find_fertilizer_composition(name, formula)
        
        if not composition_data:
            return None
        
        return Fertilizer(
            name=name,
            percentage=98.0,  # Default purity
            molecular_weight=composition_data['mw'],
            salt_weight=composition_data['mw'],
            density=1.0,  # Default density
            chemistry=FertilizerChemistry(
                formula=composition_data['formula'],
                purity=98.0,
                solubility=100.0,
                is_ph_adjuster=False
            ),
            composition=FertilizerComposition(
                cations=composition_data['cations'],
                anions=composition_data['anions']
            )
        )

    def get_complete_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive database information with complete micronutrient analysis
        """
        fertilizer_list = []
        validation_errors = []
        
        # Enhanced statistics counters
        fertilizer_types = {
            'acids': 0,
            'nitrates': 0,
            'sulfates': 0,
            'phosphates': 0,
            'chlorides': 0,
            'iron_sources': 0,
            'manganese_sources': 0,
            'zinc_sources': 0,
            'copper_sources': 0,
            'boron_sources': 0,
            'molybdenum_sources': 0,
            'micronutrient_mixes': 0,
            'chelated_forms': 0,
            'specialized_fertilizers': 0
        }
        
        for fert_key, fert_data in self.fertilizer_data.items():
            composition = fert_data['composition']
            
            # Calculate total content
            total_content = sum(composition['cations'].values()) + sum(composition['anions'].values())
            
            # Enhanced type determination
            fert_type = 'other'
            if 'acido' in fert_key and 'borico' not in fert_key:
                fert_type = 'acids'
                fertilizer_types['acids'] += 1
            elif 'nitrato' in fert_key:
                fert_type = 'nitrates'
                fertilizer_types['nitrates'] += 1
            elif 'sulfato' in fert_key:
                fert_type = 'sulfates'
                fertilizer_types['sulfates'] += 1
            elif 'fosfato' in fert_key:
                fert_type = 'phosphates'
                fertilizer_types['phosphates'] += 1
            elif 'cloruro' in fert_key:
                fert_type = 'chlorides'
                fertilizer_types['chlorides'] += 1
            elif 'quelato' in fert_key:
                fert_type = 'chelated_forms'
                fertilizer_types['chelated_forms'] += 1
            elif 'mix' in fert_key or 'cocktail' in fert_key:
                fert_type = 'micronutrient_mixes'
                fertilizer_types['micronutrient_mixes'] += 1
            
            # Count micronutrient sources by element
            if composition['cations'].get('Fe', 0) > 0.5:
                fertilizer_types['iron_sources'] += 1
            if composition['cations'].get('Mn', 0) > 0.5:
                fertilizer_types['manganese_sources'] += 1
            if composition['cations'].get('Zn', 0) > 0.5:
                fertilizer_types['zinc_sources'] += 1
            if composition['cations'].get('Cu', 0) > 0.5:
                fertilizer_types['copper_sources'] += 1
            if composition['anions'].get('B', 0) > 0.5:
                fertilizer_types['boron_sources'] += 1
            if composition['anions'].get('Mo', 0) > 0.5:
                fertilizer_types['molybdenum_sources'] += 1
            
            # Specialized fertilizers
            if any(keyword in fert_key for keyword in ['boro', 'micro', 'plus']):
                fertilizer_types['specialized_fertilizers'] += 1
            
            # Find main nutrients
            main_nutrients = []
            for elem, content in composition['cations'].items():
                if content > 1:
                    main_nutrients.append(f"{elem}:{content:.1f}%")
            for elem, content in composition['anions'].items():
                if content > 1:
                    main_nutrients.append(f"{elem}:{content:.1f}%")
            
            # Enhanced validation
            if total_content < 5:
                validation_errors.append(f"{fert_key}: Very low total content ({total_content:.1f}%)")
            if composition['mw'] <= 0:
                validation_errors.append(f"{fert_key}: Invalid molecular weight")
            
            # Check micronutrient content validity
            micronutrient_content = sum(composition['cations'].get(micro, 0) for micro in ['Fe', 'Mn', 'Zn', 'Cu']) + \
                                  sum(composition['anions'].get(micro, 0) for micro in ['B', 'Mo'])
            
            fertilizer_info = {
                'key': fert_key,
                'name': fert_data['patterns'][0].title(),
                'formula': composition['formula'],
                'molecular_weight': composition['mw'],
                'total_content': total_content,
                'micronutrient_content': micronutrient_content,
                'type': fert_type,
                'main_nutrients': main_nutrients,
                'patterns_count': len(fert_data['patterns']),
                'formula_patterns_count': len(fert_data['formula_patterns']),
                'is_micronutrient_source': micronutrient_content > 0.5
            }
            
            fertilizer_list.append(fertilizer_info)
        
        # Sort by total content (descending)
        fertilizer_list.sort(key=lambda x: x['total_content'], reverse=True)
        
        return {
            'total_fertilizers': len(self.fertilizer_data),
            'fertilizers': fertilizer_list,
            'fertilizers_by_type': fertilizer_types,
            'micronutrient_coverage': {
                'iron_sources': fertilizer_types['iron_sources'],
                'manganese_sources': fertilizer_types['manganese_sources'],
                'zinc_sources': fertilizer_types['zinc_sources'],
                'copper_sources': fertilizer_types['copper_sources'],
                'boron_sources': fertilizer_types['boron_sources'],
                'molybdenum_sources': fertilizer_types['molybdenum_sources'],
                'total_micronutrient_sources': sum([
                    fertilizer_types['iron_sources'],
                    fertilizer_types['manganese_sources'],
                    fertilizer_types['zinc_sources'],
                    fertilizer_types['copper_sources'],
                    fertilizer_types['boron_sources'],
                    fertilizer_types['molybdenum_sources']
                ])
            },
            'validation_report': {
                'validation_errors': validation_errors,
                'statistics': {
                    'average_content': sum(f['total_content'] for f in fertilizer_list) / len(fertilizer_list),
                    'fertilizers_by_type': fertilizer_types,
                    'pattern_coverage': sum(f['patterns_count'] for f in fertilizer_list),
                    'formula_coverage': sum(f['formula_patterns_count'] for f in fertilizer_list),
                    'micronutrient_fertilizers': len([f for f in fertilizer_list if f['is_micronutrient_source']])
                }
            },
            'database_status': 'enhanced_with_complete_micronutrients',
            'coverage': {
                'macronutrients': 'complete',
                'micronutrients': 'complete',
                'ph_adjusters': 'complete',
                'specialty_fertilizers': 'complete',
                'chelated_forms': 'complete'
            },
            'new_micronutrient_features': [
                f"Iron sources: {fertilizer_types['iron_sources']} (chelates, sulfates, chlorides)",
                f"Manganese sources: {fertilizer_types['manganese_sources']} (sulfates, chlorides, chelates)",
                f"Zinc sources: {fertilizer_types['zinc_sources']} (sulfates, chlorides, chelates)",
                f"Copper sources: {fertilizer_types['copper_sources']} (sulfates, chlorides, chelates)",
                f"Boron sources: {fertilizer_types['boron_sources']} (boric acid, borax, soluboro)",
                f"Molybdenum sources: {fertilizer_types['molybdenum_sources']} (sodium, ammonium, calcium molybdate)",
                f"Micronutrient mixes: {fertilizer_types['micronutrient_mixes']} professional blends",
                f"Chelated forms: {fertilizer_types['chelated_forms']} EDTA/DTPA chelates",
                f"Specialized fertilizers: {fertilizer_types['specialized_fertilizers']} enhanced formulations"
            ]
        }

    def find_fertilizers_containing_element(self, element: str, min_content: float = 0.1) -> List[Dict[str, Any]]:
        """
        Find all fertilizers containing a specific element above minimum content (enhanced for micronutrients)
        """
        element = element.upper()
        matching_fertilizers = []
        
        for fert_key, fert_data in self.fertilizer_data.items():
            composition = fert_data['composition']
            
            # Check both cations and anions
            cation_content = composition['cations'].get(element, 0)
            anion_content = composition['anions'].get(element, 0)
            total_element_content = cation_content + anion_content
            
            if total_element_content >= min_content:
                # Determine source type and quality
                source_type = 'cation' if cation_content > anion_content else 'anion'
                
                # Enhanced quality assessment for micronutrients
                quality_rating = 'standard'
                if 'quelato' in fert_key or 'chelate' in fert_key:
                    quality_rating = 'premium_chelated'
                elif element in ['FE', 'MN', 'ZN', 'CU'] and total_element_content > 15:
                    quality_rating = 'high_concentration'
                elif 'mix' in fert_key:
                    quality_rating = 'balanced_blend'
                
                matching_fertilizers.append({
                    'name': fert_data['patterns'][0].title(),
                    'formula': composition['formula'],
                    'element_content': total_element_content,
                    'molecular_weight': composition['mw'],
                    'source_type': source_type,
                    'quality_rating': quality_rating,
                    'patterns': fert_data['patterns'],
                    'is_micronutrient_specific': any(micro in fert_key for micro in ['hierro', 'manganeso', 'zinc', 'cobre', 'borico', 'molibdato'])
                })
        
        # Sort by element content (descending) and then by quality
        quality_order = {'premium_chelated': 4, 'high_concentration': 3, 'balanced_blend': 2, 'standard': 1}
        matching_fertilizers.sort(key=lambda x: (x['element_content'], quality_order.get(x['quality_rating'], 0)), reverse=True)
        
        return matching_fertilizers

    def get_micronutrient_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get professional micronutrient recommendations for different scenarios
        """
        return {
            'water_quality_recommendations': {
                'soft_water': {
                    'description': 'Low mineral content water (EC < 0.5 dS/m)',
                    'iron_recommendation': 'Fe-EDTA chelate for stability',
                    'manganese_recommendation': 'MnSO4 - good solubility',
                    'zinc_recommendation': 'ZnSO4 - cost effective',
                    'copper_recommendation': 'CuSO4 - adequate availability',
                    'boron_recommendation': 'Boric acid - precise control',
                    'molybdenum_recommendation': 'Sodium molybdate - high solubility'
                },
                'medium_water': {
                    'description': 'Moderate mineral content (EC 0.5-1.5 dS/m)',
                    'iron_recommendation': 'Fe-EDTA or Fe-DTPA chelate',
                    'manganese_recommendation': 'Mn-EDTA for better availability',
                    'zinc_recommendation': 'Zn-EDTA in alkaline conditions',
                    'copper_recommendation': 'Cu-EDTA or CuSO4',
                    'boron_recommendation': 'Boric acid or Soluboro',
                    'molybdenum_recommendation': 'Sodium or ammonium molybdate'
                },
                'hard_water': {
                    'description': 'High mineral content (EC > 1.5 dS/m)',
                    'iron_recommendation': 'Fe-DTPA chelate mandatory',
                    'manganese_recommendation': 'Mn-EDTA chelate required',
                    'zinc_recommendation': 'Zn-EDTA chelate essential',
                    'copper_recommendation': 'Cu-EDTA chelate preferred',
                    'boron_recommendation': 'Soluboro for better availability',
                    'molybdenum_recommendation': 'Calcium molybdate for stability'
                }
            },
            'crop_specific_recommendations': {
                'leafy_greens': {
                    'iron': {'range': '1.5-2.5 mg/L', 'preferred_source': 'Fe-EDTA'},
                    'manganese': {'range': '0.4-0.7 mg/L', 'preferred_source': 'MnSO4'},
                    'zinc': {'range': '0.2-0.4 mg/L', 'preferred_source': 'ZnSO4'},
                    'copper': {'range': '0.08-0.15 mg/L', 'preferred_source': 'CuSO4'},
                    'boron': {'range': '0.3-0.6 mg/L', 'preferred_source': 'Boric acid'},
                    'molybdenum': {'range': '0.03-0.07 mg/L', 'preferred_source': 'Sodium molybdate'}
                },
                'fruiting_crops': {
                    'iron': {'range': '2.0-3.5 mg/L', 'preferred_source': 'Fe-EDTA'},
                    'manganese': {'range': '0.5-1.0 mg/L', 'preferred_source': 'Mn-EDTA'},
                    'zinc': {'range': '0.3-0.6 mg/L', 'preferred_source': 'Zn-EDTA'},
                    'copper': {'range': '0.1-0.2 mg/L', 'preferred_source': 'Cu-EDTA'},
                    'boron': {'range': '0.4-0.8 mg/L', 'preferred_source': 'Borax'},
                    'molybdenum': {'range': '0.04-0.08 mg/L', 'preferred_source': 'Ammonium molybdate'}
                },
                'herbs': {
                    'iron': {'range': '1.2-2.2 mg/L', 'preferred_source': 'Fe-EDTA'},
                    'manganese': {'range': '0.3-0.6 mg/L', 'preferred_source': 'MnSO4'},
                    'zinc': {'range': '0.15-0.35 mg/L', 'preferred_source': 'ZnSO4'},
                    'copper': {'range': '0.06-0.12 mg/L', 'preferred_source': 'CuSO4'},
                    'boron': {'range': '0.25-0.55 mg/L', 'preferred_source': 'Boric acid'},
                    'molybdenum': {'range': '0.02-0.06 mg/L', 'preferred_source': 'Sodium molybdate'}
                },
                'root_vegetables': {
                    'iron': {'range': '1.8-2.8 mg/L', 'preferred_source': 'Fe-DTPA'},
                    'manganese': {'range': '0.6-1.2 mg/L', 'preferred_source': 'MnSO4'},
                    'zinc': {'range': '0.25-0.45 mg/L', 'preferred_source': 'ZnSO4'},
                    'copper': {'range': '0.09-0.18 mg/L', 'preferred_source': 'CuSO4'},
                    'boron': {'range': '0.5-1.0 mg/L', 'preferred_source': 'Borax'},
                    'molybdenum': {'range': '0.03-0.07 mg/L', 'preferred_source': 'Sodium molybdate'}
                }
            },
            'application_guidelines': {
                'stock_solution_preparation': {
                    'micronutrient_mix': 'Prepare separate micronutrient stock (1000x concentration)',
                    'storage': 'Store in dark, cool conditions (<25°C)',
                    'stability': 'Use within 2-4 weeks for optimal effectiveness',
                    'mixing_order': 'Add micronutrients last to prevent precipitation'
                },
                'chelate_selection': {
                    'ph_considerations': {
                        'ph_below_6': 'Sulfate forms generally adequate',
                        'ph_6_to_7': 'EDTA chelates recommended',
                        'ph_above_7': 'DTPA chelates essential for Fe, Zn'
                    },
                    'cost_vs_performance': {
                        'budget_option': 'Sulfate forms with pH management',
                        'standard_option': 'EDTA chelates for most applications',
                        'premium_option': 'DTPA chelates for challenging conditions'
                    }
                },
                'troubleshooting': {
                    'iron_deficiency_symptoms': 'Yellowing between leaf veins (interveinal chlorosis)',
                    'manganese_deficiency': 'Small yellow/brown spots on leaves',
                    'zinc_deficiency': 'Stunted growth, small leaves, short internodes',
                    'copper_deficiency': 'Wilting tips, blue-green coloration',
                    'boron_deficiency': 'Brittle leaves, poor fruit development',
                    'molybdenum_deficiency': 'Yellow leaves similar to nitrogen deficiency'
                }
            },
            'dosage_recommendations': {
                'micronutrient_ratios': {
                    'fe_mn_ratio': '4:1 to 6:1 (Fe:Mn)',
                    'fe_zn_ratio': '6:1 to 10:1 (Fe:Zn)',
                    'mn_zn_ratio': '1.5:1 to 2:1 (Mn:Zn)',
                    'cu_zn_ratio': '1:3 to 1:4 (Cu:Zn)'
                },
                'safety_limits': {
                    'iron_max': '5.0 mg/L (above this may cause toxicity)',
                    'manganese_max': '2.0 mg/L (toxic at high levels)',
                    'zinc_max': '1.0 mg/L (inhibits other nutrients)',
                    'copper_max': '0.5 mg/L (very toxic when excessive)',
                    'boron_max': '1.5 mg/L (narrow optimal range)',
                    'molybdenum_max': '0.2 mg/L (can interfere with copper)'
                }
            }
        }

    def get_complete_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive database information including micronutrients
        """
        fertilizer_list = []
        validation_errors = []
        
        # Statistics counters
        fertilizer_types = {
            'acids': 0,
            'nitrates': 0,
            'sulfates': 0,
            'phosphates': 0,
            'chlorides': 0,
            'iron_sources': 0,
            'manganese_sources': 0,
            'zinc_sources': 0,
            'copper_sources': 0,
            'boron_sources': 0,
            'molybdenum_sources': 0,
            'chelates': 0,
            'micronutrient_mixes': 0
        }
        
        # Micronutrient coverage tracking
        micronutrient_coverage = {
            'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0, 'B': 0, 'Mo': 0
        }
        
        for fert_key, fert_data in self.fertilizer_data.items():
            composition = fert_data['composition']
            
            # Calculate total content
            total_content = sum(composition['cations'].values()) + sum(composition['anions'].values())
            
            # Determine type and count micronutrient sources
            fert_type = 'other'
            if 'acido' in fert_key and 'borico' not in fert_key:
                fert_type = 'acids'
                fertilizer_types['acids'] += 1
            elif 'nitrato' in fert_key:
                fert_type = 'nitrates'
                fertilizer_types['nitrates'] += 1
            elif 'sulfato' in fert_key:
                if any(elem in fert_key for elem in ['hierro', 'manganeso', 'zinc', 'cobre']):
                    fert_type = 'micronutrient_sulfates'
                    if 'hierro' in fert_key:
                        fertilizer_types['iron_sources'] += 1
                    elif 'manganeso' in fert_key:
                        fertilizer_types['manganese_sources'] += 1
                    elif 'zinc' in fert_key:
                        fertilizer_types['zinc_sources'] += 1
                    elif 'cobre' in fert_key:
                        fertilizer_types['copper_sources'] += 1
                else:
                    fert_type = 'sulfates'
                    fertilizer_types['sulfates'] += 1
            elif 'fosfato' in fert_key:
                fert_type = 'phosphates'
                fertilizer_types['phosphates'] += 1
            elif 'cloruro' in fert_key:
                if any(elem in fert_key for elem in ['hierro', 'manganeso', 'zinc', 'cobre']):
                    fert_type = 'micronutrient_chlorides'
                    if 'hierro' in fert_key:
                        fertilizer_types['iron_sources'] += 1
                    elif 'manganeso' in fert_key:
                        fertilizer_types['manganese_sources'] += 1
                    elif 'zinc' in fert_key:
                        fertilizer_types['zinc_sources'] += 1
                    elif 'cobre' in fert_key:
                        fertilizer_types['copper_sources'] += 1
                else:
                    fert_type = 'chlorides'
                    fertilizer_types['chlorides'] += 1
            elif 'quelato' in fert_key:
                fert_type = 'chelates'
                fertilizer_types['chelates'] += 1
                if 'hierro' in fert_key:
                    fertilizer_types['iron_sources'] += 1
                elif 'manganeso' in fert_key:
                    fertilizer_types['manganese_sources'] += 1
                elif 'zinc' in fert_key:
                    fertilizer_types['zinc_sources'] += 1
                elif 'cobre' in fert_key:
                    fertilizer_types['copper_sources'] += 1
            elif any(elem in fert_key for elem in ['borico', 'borax']):
                fert_type = 'boron_sources'
                fertilizer_types['boron_sources'] += 1
            elif 'molibdato' in fert_key:
                fert_type = 'molybdenum_sources'
                fertilizer_types['molybdenum_sources'] += 1
            elif 'mix' in fert_key:
                fert_type = 'micronutrient_mixes'
                fertilizer_types['micronutrient_mixes'] += 1
            
            # Count micronutrient sources
            for micro in micronutrient_coverage.keys():
                content = composition['cations'].get(micro, 0) + composition['anions'].get(micro, 0)
                if content > 0.1:
                    micronutrient_coverage[micro] += 1
            
            # Find main nutrients
            main_nutrients = []
            for elem, content in composition['cations'].items():
                if content > 1:
                    main_nutrients.append(f"{elem}:{content:.1f}%")
            for elem, content in composition['anions'].items():
                if content > 1:
                    main_nutrients.append(f"{elem}:{content:.1f}%")
            
            # Validation
            if total_content < 5:
                validation_errors.append(f"{fert_key}: Very low total content ({total_content:.1f}%)")
            if composition['mw'] <= 0:
                validation_errors.append(f"{fert_key}: Invalid molecular weight")
            
            fertilizer_info = {
                'key': fert_key,
                'name': fert_data['patterns'][0].title(),
                'formula': composition['formula'],
                'molecular_weight': composition['mw'],
                'total_content': total_content,
                'type': fert_type,
                'main_nutrients': main_nutrients,
                'patterns_count': len(fert_data['patterns']),
                'formula_patterns_count': len(fert_data['formula_patterns']),
                'micronutrient_content': {
                    micro: composition['cations'].get(micro, 0) + composition['anions'].get(micro, 0)
                    for micro in ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
                    if composition['cations'].get(micro, 0) + composition['anions'].get(micro, 0) > 0
                }
            }
            
            fertilizer_list.append(fertilizer_info)
        
        # Sort by total content (descending)
        fertilizer_list.sort(key=lambda x: x['total_content'], reverse=True)
        
        return {
            'total_fertilizers': len(self.fertilizer_data),
            'fertilizers': fertilizer_list,
            'fertilizers_by_type': fertilizer_types,
            'micronutrient_coverage': micronutrient_coverage,
            'micronutrient_sources': self.get_micronutrient_sources(),
            'validation_report': {
                'validation_errors': validation_errors,
                'statistics': {
                    'average_content': sum(f['total_content'] for f in fertilizer_list) / len(fertilizer_list),
                    'fertilizers_by_type': fertilizer_types,
                    'pattern_coverage': sum(f['patterns_count'] for f in fertilizer_list),
                    'formula_coverage': sum(f['formula_patterns_count'] for f in fertilizer_list),
                    'micronutrient_fertilizers': sum(1 for f in fertilizer_list if f['micronutrient_content'])
                }
            },
            'database_status': 'complete_with_micronutrients',
            'coverage': {
                'macronutrients': 'complete',
                'micronutrients': 'complete',
                'ph_adjusters': 'complete',
                'specialty_fertilizers': 'complete',
                'chelated_forms': 'complete'
            },
            'micronutrient_features': {
                'iron_sources': fertilizer_types['iron_sources'],
                'manganese_sources': fertilizer_types['manganese_sources'],
                'zinc_sources': fertilizer_types['zinc_sources'],
                'copper_sources': fertilizer_types['copper_sources'],
                'boron_sources': fertilizer_types['boron_sources'],
                'molybdenum_sources': fertilizer_types['molybdenum_sources'],
                'chelated_forms': fertilizer_types['chelates'],
                'total_micronutrient_fertilizers': sum([
                    fertilizer_types['iron_sources'],
                    fertilizer_types['manganese_sources'],
                    fertilizer_types['zinc_sources'],
                    fertilizer_types['copper_sources'],
                    fertilizer_types['boron_sources'],
                    fertilizer_types['molybdenum_sources']
                ])
            }
        }

    def get_micronutrient_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available sources for each micronutrient with detailed information"""
        micronutrient_sources = {
            'Fe': [],
            'Mn': [],
            'Zn': [],
            'Cu': [],
            'B': [],
            'Mo': []
        }
        
        for fert_key, fert_data in self.fertilizer_data.items():
            composition = fert_data['composition']
            for micronutrient in micronutrient_sources.keys():
                content = composition['cations'].get(micronutrient, 0) + composition['anions'].get(micronutrient, 0)
                if content > 0.05:  # Include sources with >0.05% content
                    
                    # Determine source type
                    source_type = 'sulfate'
                    if 'quelato' in fert_key or 'edta' in fert_key or 'dtpa' in fert_key:
                        source_type = 'chelate'
                    elif 'cloruro' in fert_key:
                        source_type = 'chloride'
                    elif 'borico' in fert_key:
                        source_type = 'acid'
                    elif 'molibdato' in fert_key:
                        source_type = 'molybdate'
                    elif 'borax' in fert_key:
                        source_type = 'borate'
                    
                    # Determine water suitability
                    water_suitability = []
                    if source_type == 'chelate':
                        water_suitability = ['soft', 'medium', 'hard']
                    elif source_type in ['sulfate', 'chloride']:
                        if micronutrient in ['Fe', 'Mn', 'Zn']:
                            water_suitability = ['soft', 'medium']
                        else:
                            water_suitability = ['soft', 'medium', 'hard']
                    else:
                        water_suitability = ['soft', 'medium', 'hard']
                    
                    micronutrient_sources[micronutrient].append({
                        'name': fert_data['patterns'][0].title(),
                        'formula': composition['formula'],
                        'content': content,
                        'source_type': source_type,
                        'molecular_weight': composition['mw'],
                        'water_suitability': water_suitability,
                        'relative_cost': self._estimate_relative_cost(fert_key, source_type),
                        'stability': self._estimate_stability(source_type),
                        'recommended_for': self._get_crop_recommendations(micronutrient, source_type)
                    })
        
        # Sort by content (highest first) for each micronutrient
        for micronutrient in micronutrient_sources:
            micronutrient_sources[micronutrient].sort(key=lambda x: x['content'], reverse=True)
        
        return micronutrient_sources

    def _estimate_relative_cost(self, fert_key: str, source_type: str) -> str:
        """Estimate relative cost of fertilizer source"""
        if source_type == 'chelate':
            return 'high'
        elif 'molibdato' in fert_key:
            return 'high'
        elif source_type in ['sulfate', 'chloride']:
            return 'low'
        elif 'borico' in fert_key:
            return 'medium'
        else:
            return 'medium'

    def _estimate_stability(self, source_type: str) -> str:
        """Estimate stability of fertilizer source"""
        if source_type == 'chelate':
            return 'high'
        elif source_type in ['sulfate', 'chloride']:
            return 'medium'
        elif source_type in ['acid', 'molybdate', 'borate']:
            return 'high'
        else:
            return 'medium'

    def _get_crop_recommendations(self, micronutrient: str, source_type: str) -> List[str]:
        """Get crop recommendations for specific micronutrient sources"""
        recommendations = []
        
        if source_type == 'chelate':
            recommendations = ['all_crops', 'hard_water', 'high_ph']
        elif source_type == 'sulfate':
            if micronutrient in ['Fe', 'Mn', 'Zn']:
                recommendations = ['soft_water', 'acidic_conditions', 'cost_sensitive']
            else:
                recommendations = ['general_use', 'cost_effective']
        elif source_type == 'chloride':
            recommendations = ['chloride_tolerant_crops', 'short_term_use']
        elif source_type in ['acid', 'molybdate', 'borate']:
            recommendations = ['precise_dosing', 'all_water_types']
        
        return recommendations

    def find_fertilizers_containing_element(self, element: str, min_content: float = 0.1) -> List[Dict[str, Any]]:
        """
        Find all fertilizers containing a specific element above minimum content
        """
        element = element.upper()
        matching_fertilizers = []
        
        for fert_key, fert_data in self.fertilizer_data.items():
            composition = fert_data['composition']
            
            # Check both cations and anions
            cation_content = composition['cations'].get(element, 0)
            anion_content = composition['anions'].get(element, 0)
            total_element_content = cation_content + anion_content
            
            if total_element_content >= min_content:
                # Determine if this is a micronutrient fertilizer
                is_micronutrient_fertilizer = element in ['FE', 'MN', 'ZN', 'CU', 'B', 'MO']
                
                # Get source type
                source_type = 'other'
                if 'quelato' in fert_key:
                    source_type = 'chelate'
                elif 'sulfato' in fert_key:
                    source_type = 'sulfate'
                elif 'cloruro' in fert_key:
                    source_type = 'chloride'
                elif 'nitrato' in fert_key:
                    source_type = 'nitrate'
                elif 'fosfato' in fert_key:
                    source_type = 'phosphate'
                elif 'borico' in fert_key:
                    source_type = 'acid'
                elif 'molibdato' in fert_key:
                    source_type = 'molybdate'
                
                matching_fertilizers.append({
                    'name': fert_data['patterns'][0].title(),
                    'formula': composition['formula'],
                    'element_content': total_element_content,
                    'molecular_weight': composition['mw'],
                    'source_type': 'cation' if cation_content > anion_content else 'anion',
                    'fertilizer_type': source_type,
                    'is_micronutrient_fertilizer': is_micronutrient_fertilizer,
                    'patterns': fert_data['patterns'],
                    'recommended_use': self._get_element_use_recommendations(element, source_type, total_element_content)
                })
        
        # Sort by element content (descending)
        matching_fertilizers.sort(key=lambda x: x['element_content'], reverse=True)
        
        return matching_fertilizers

    def _get_element_use_recommendations(self, element: str, source_type: str, content: float) -> Dict[str, Any]:
        """Get usage recommendations for specific element and source type"""
        recommendations = {
            'application_rate': 'standard',
            'water_compatibility': 'medium',
            'storage_requirements': 'cool_dry',
            'mixing_instructions': 'add_slowly',
            'special_considerations': []
        }
        
        # Element-specific recommendations
        if element == 'FE':
            if source_type == 'chelate':
                recommendations['water_compatibility'] = 'excellent'
                recommendations['special_considerations'].append('Suitable for all pH ranges')
            elif source_type == 'sulfate':
                recommendations['water_compatibility'] = 'good_acidic'
                recommendations['special_considerations'].append('Best in acidic conditions (pH < 6.5)')
                
        elif element in ['MN', 'ZN']:
            if source_type == 'chelate':
                recommendations['water_compatibility'] = 'excellent'
            elif source_type == 'sulfate':
                recommendations['water_compatibility'] = 'good'
                recommendations['special_considerations'].append('Monitor pH for optimal availability')
                
        elif element == 'CU':
            recommendations['application_rate'] = 'low'
            recommendations['special_considerations'].append('Highly toxic in excess - use carefully')
            
        elif element == 'B':
            recommendations['application_rate'] = 'precise'
            recommendations['special_considerations'].append('Narrow range between deficiency and toxicity')
            
        elif element == 'MO':
            recommendations['application_rate'] = 'very_low'
            recommendations['special_considerations'].append('Required in trace amounts only')
        
        # High-content sources need more careful dosing
        if content > 20:
            recommendations['application_rate'] = 'low'
            recommendations['special_considerations'].append('High concentration - dilute carefully')
        
        return recommendations

    def validate_database_integrity(self) -> Dict[str, Any]:
        """
        Validate database integrity and completeness including micronutrients
        """
        validation_results = {
            'total_entries': len(self.fertilizer_data),
            'validation_errors': [],
            'warnings': [],
            'coverage_analysis': {},
            'micronutrient_analysis': {},
            'integrity_score': 0
        }
        
        required_elements = ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        element_coverage = {elem: 0 for elem in required_elements}
        
        # Micronutrient-specific validation
        micronutrient_requirements = {
            'Fe': {'min_sources': 3, 'min_chelates': 1, 'found_sources': 0, 'found_chelates': 0},
            'Mn': {'min_sources': 2, 'min_chelates': 1, 'found_sources': 0, 'found_chelates': 0},
            'Zn': {'min_sources': 2, 'min_chelates': 1, 'found_sources': 0, 'found_chelates': 0},
            'Cu': {'min_sources': 2, 'min_chelates': 1, 'found_sources': 0, 'found_chelates': 0},
            'B': {'min_sources': 2, 'min_chelates': 0, 'found_sources': 0, 'found_chelates': 0},
            'Mo': {'min_sources': 2, 'min_chelates': 0, 'found_sources': 0, 'found_chelates': 0}
        }
        
        for fert_key, fert_data in self.fertilizer_data.items():
            composition = fert_data['composition']
            
            # Check molecular weight
            if composition['mw'] <= 0:
                validation_results['validation_errors'].append(f"{fert_key}: Invalid molecular weight")
            
            # Check total content
            total_content = sum(composition['cations'].values()) + sum(composition['anions'].values())
            if total_content < 5:
                validation_results['warnings'].append(f"{fert_key}: Low total content ({total_content:.1f}%)")
            elif total_content > 120:  # Allow higher for micronutrient concentrated sources
                validation_results['warnings'].append(f"{fert_key}: Very high content ({total_content:.1f}%)")
            
            # Check element coverage and micronutrient sources
            for element in required_elements:
                cation_content = composition['cations'].get(element, 0)
                anion_content = composition['anions'].get(element, 0)
                total_element_content = cation_content + anion_content
                
                if total_element_content > 1:  # Significant content
                    element_coverage[element] += 1
                    
                    # Track micronutrient sources
                    if element in micronutrient_requirements:
                        micronutrient_requirements[element]['found_sources'] += 1
                        if 'quelato' in fert_key or 'edta' in fert_key or 'dtpa' in fert_key:
                            micronutrient_requirements[element]['found_chelates'] += 1
        
        # Analyze micronutrient coverage
        micronutrient_issues = []
        for micro, req in micronutrient_requirements.items():
            if req['found_sources'] < req['min_sources']:
                micronutrient_issues.append(f"{micro}: Only {req['found_sources']} sources (need {req['min_sources']})")
            if req['found_chelates'] < req['min_chelates']:
                micronutrient_issues.append(f"{micro}: Only {req['found_chelates']} chelates (need {req['min_chelates']})")
        
        # Calculate coverage analysis
        validation_results['coverage_analysis'] = {
            'element_coverage': element_coverage,
            'micronutrient_issues': micronutrient_issues
        }
        
        # Calculate integrity score (simple heuristic)
        validation_results['integrity_score'] = 100 - len(validation_results['validation_errors']) * 10
        if validation_results['integrity_score'] < 0:
            validation_results['integrity_score'] = 0
        
        return validation_results