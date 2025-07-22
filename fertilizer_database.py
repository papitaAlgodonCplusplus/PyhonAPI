#!/usr/bin/env python3
"""
COMPLETE FERTILIZER DATABASE MODULE
Comprehensive fertilizer composition database with intelligent matching
"""

from typing import Dict, Optional, List, Any
from models import Fertilizer, FertilizerComposition, FertilizerChemistry

class FertilizerDatabase:
    """Complete fertilizer composition database with intelligent pattern matching"""
    
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
            
            # MICRONUTRIENT SOURCES
            'quelato_hierro': {
                'patterns': ['quelato de hierro', 'iron chelate', 'fe-edta', 'fe edta', 'feeedta'],
                'formula_patterns': ['FE-EDTA', 'C10H12FeN2NaO8'],
                'composition': {
                    'formula': 'C10H12FeN2NaO8',
                    'mw': 367.05,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 6.27, 'NH4': 0, 'Fe': 15.22, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 7.63, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'sulfato_hierro': {
                'patterns': ['sulfato de hierro', 'iron sulfate', 'sulfato ferroso', 'feso4'],
                'formula_patterns': ['FESO4', 'FESO4.7H2O', 'FeSO4', 'FeSO4.7H2O'],
                'composition': {
                    'formula': 'FeSO4.7H2O',
                    'mw': 278.01,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 20.09, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 11.53, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'sulfato_manganeso': {
                'patterns': ['sulfato de manganeso', 'manganese sulfate', 'sulfato manganeso', 'mnso4'],
                'formula_patterns': ['MNSO4', 'MNSO4.4H2O', 'MnSO4', 'MnSO4.4H2O'],
                'composition': {
                    'formula': 'MnSO4.4H2O',
                    'mw': 223.06,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 24.63, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 14.37, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'sulfato_zinc': {
                'patterns': ['sulfato de zinc', 'zinc sulfate', 'sulfato zinc', 'znso4'],
                'formula_patterns': ['ZNSO4', 'ZNSO4.7H2O', 'ZnSO4', 'ZnSO4.7H2O'],
                'composition': {
                    'formula': 'ZnSO4.7H2O',
                    'mw': 287.56,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 22.74, 'Cu': 0},
                    'anions': {'N': 0, 'S': 11.15, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'sulfato_cobre': {
                'patterns': ['sulfato de cobre', 'copper sulfate', 'sulfato cobre', 'cuso4'],
                'formula_patterns': ['CUSO4', 'CUSO4.5H2O', 'CuSO4', 'CuSO4.5H2O'],
                'composition': {
                    'formula': 'CuSO4.5H2O',
                    'mw': 249.69,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 25.45},
                    'anions': {'N': 0, 'S': 12.84, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'acido_borico': {
                'patterns': ['acido borico', 'ácido bórico', 'boric acid', 'h3bo3'],
                'formula_patterns': ['H3BO3'],
                'composition': {
                    'formula': 'H3BO3',
                    'mw': 61.83,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 17.48, 'Mo': 0}
                }
            },
            'molibdato_sodio': {
                'patterns': ['molibdato de sodio', 'sodium molybdate', 'molibdato sodio', 'na2moo4'],
                'formula_patterns': ['NA2MOO4', 'NA2MOO4.2H2O', 'Na2MoO4', 'Na2MoO4.2H2O'],
                'composition': {
                    'formula': 'Na2MoO4.2H2O',
                    'mw': 241.95,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 19.01, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 39.66}
                }
            }
        }

    def find_fertilizer_composition(self, name: str, formula: str = "") -> Optional[Dict[str, Any]]:
        """
        Find fertilizer composition by intelligent name and formula matching
        """
        name_lower = name.lower().strip()
        formula_upper = formula.upper().strip()

        print(f"    Searching database for: name='{name_lower}', formula='{formula_upper}'")

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

        # Strategy 3: Fuzzy matching for common variations
        fuzzy_matches = {
            'calcium': 'nitrato_calcio',
            'potassium': 'nitrato_potasio',
            'nitrate': 'nitrato_calcio',
            'phosphate': 'fosfato_monopotasico',
            'sulfate': 'sulfato_magnesio',
            'magnesium': 'sulfato_magnesio',
            'iron': 'quelato_hierro',
            'manganese': 'sulfato_manganeso',
            'zinc': 'sulfato_zinc',
            'copper': 'sulfato_cobre',
            'boron': 'acido_borico',
            'molybdenum': 'molibdato_sodio'
        }
        
        for keyword, fert_key in fuzzy_matches.items():
            if keyword in name_lower:
                if fert_key in self.fertilizer_data:
                    print(f"    [FOUND] by fuzzy matching: '{keyword}' -> {self.fertilizer_data[fert_key]['composition']['formula']}")
                    return self.fertilizer_data[fert_key]['composition']

        print(f"    [NOT FOUND] NO MATCH FOUND for '{name}' with formula '{formula}'")
        return None

    def create_fertilizer_from_database(self, name: str, formula: str = "") -> Optional[Fertilizer]:
        """
        Create a complete Fertilizer object from database
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
        Get comprehensive database information
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
            'micronutrients': 0
        }
        
        for fert_key, fert_data in self.fertilizer_data.items():
            composition = fert_data['composition']
            
            # Calculate total content
            total_content = sum(composition['cations'].values()) + sum(composition['anions'].values())
            
            # Determine type
            fert_type = 'other'
            if 'acido' in fert_key:
                fert_type = 'acids'
                fertilizer_types['acids'] += 1
            elif 'nitrato' in fert_key:
                fert_type = 'nitrates'
                fertilizer_types['nitrates'] += 1
            elif 'sulfato' in fert_key:
                if any(elem in fert_key for elem in ['hierro', 'manganeso', 'zinc', 'cobre']):
                    fert_type = 'micronutrients'
                    fertilizer_types['micronutrients'] += 1
                else:
                    fert_type = 'sulfates'
                    fertilizer_types['sulfates'] += 1
            elif 'fosfato' in fert_key:
                fert_type = 'phosphates'
                fertilizer_types['phosphates'] += 1
            elif 'cloruro' in fert_key:
                fert_type = 'chlorides'
                fertilizer_types['chlorides'] += 1
            elif any(micro in fert_key for micro in ['quelato', 'borico', 'molibdato']):
                fert_type = 'micronutrients'
                fertilizer_types['micronutrients'] += 1
            
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
                'formula_patterns_count': len(fert_data['formula_patterns'])
            }
            
            fertilizer_list.append(fertilizer_info)
        
        # Sort by total content (descending)
        fertilizer_list.sort(key=lambda x: x['total_content'], reverse=True)
        
        return {
            'total_fertilizers': len(self.fertilizer_data),
            'fertilizers': fertilizer_list,
            'fertilizers_by_type': fertilizer_types,
            'validation_report': {
                'validation_errors': validation_errors,
                'statistics': {
                    'average_content': sum(f['total_content'] for f in fertilizer_list) / len(fertilizer_list),
                    'fertilizers_by_type': fertilizer_types,
                    'pattern_coverage': sum(f['patterns_count'] for f in fertilizer_list),
                    'formula_coverage': sum(f['formula_patterns_count'] for f in fertilizer_list)
                }
            },
            'database_status': 'operational',
            'coverage': {
                'macronutrients': 'complete',
                'micronutrients': 'complete',
                'ph_adjusters': 'complete',
                'specialty_fertilizers': 'partial'
            }
        }

    def find_fertilizers_containing_element(self, element: str, min_content: float = 1.0) -> List[Dict[str, Any]]:
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
                matching_fertilizers.append({
                    'name': fert_data['patterns'][0].title(),
                    'formula': composition['formula'],
                    'element_content': total_element_content,
                    'molecular_weight': composition['mw'],
                    'source_type': 'cation' if cation_content > anion_content else 'anion'
                })
        
        # Sort by element content (descending)
        matching_fertilizers.sort(key=lambda x: x['element_content'], reverse=True)
        
        return matching_fertilizers

    def get_fertilizer_suggestions(self, targets: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Get fertilizer suggestions based on target concentrations
        """
        suggestions = {}
        
        for element, target in targets.items():
            if target > 0:
                fertilizers = self.find_fertilizers_containing_element(element, min_content=1.0)
                if fertilizers:
                    # Get top 3 suggestions
                    top_suggestions = [f['name'] for f in fertilizers[:3]]
                    suggestions[element] = top_suggestions
                else:
                    suggestions[element] = ['No specific source found - use micronutrient mix']
        
        return suggestions

    def validate_database_integrity(self) -> Dict[str, Any]:
        """
        Validate database integrity and completeness
        """
        validation_results = {
            'total_entries': len(self.fertilizer_data),
            'validation_errors': [],
            'warnings': [],
            'coverage_analysis': {},
            'integrity_score': 0
        }
        
        required_elements = ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        element_coverage = {elem: 0 for elem in required_elements}
        
        for fert_key, fert_data in self.fertilizer_data.items():
            composition = fert_data['composition']
            
            # Check molecular weight
            if composition['mw'] <= 0:
                validation_results['validation_errors'].append(f"{fert_key}: Invalid molecular weight")
            
            # Check total content
            total_content = sum(composition['cations'].values()) + sum(composition['anions'].values())
            if total_content < 5:
                validation_results['warnings'].append(f"{fert_key}: Low total content ({total_content:.1f}%)")
            elif total_content > 100:
                validation_results['validation_errors'].append(f"{fert_key}: Total content exceeds 100% ({total_content:.1f}%)")
            
            # Check element coverage
            for element in required_elements:
                cation_content = composition['cations'].get(element, 0)
                anion_content = composition['anions'].get(element, 0)
                if (cation_content + anion_content) > 5:
                    element_coverage[element] += 1
        
        # Calculate coverage analysis
        validation_results['coverage_analysis'] = {
            'element_coverage': element_coverage,
            'uncovered_elements': [elem for elem, count in element_coverage.items() if count == 0],
            'well_covered_elements': [elem for elem, count in element_coverage.items() if count >= 2]
        }
        
        # Calculate integrity score
        error_penalty = len(validation_results['validation_errors']) * 10
        warning_penalty = len(validation_results['warnings']) * 2
        coverage_bonus = len(validation_results['coverage_analysis']['well_covered_elements']) * 5
        
        base_score = 100
        validation_results['integrity_score'] = max(0, base_score - error_penalty - warning_penalty + coverage_bonus)
        
        return validation_results

    def search_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search fertilizers by multiple keywords
        """
        keywords_lower = [kw.lower() for kw in keywords]
        matching_fertilizers = []
        
        for fert_key, fert_data in self.fertilizer_data.items():
            score = 0
            
            # Check patterns
            for pattern in fert_data['patterns']:
                for keyword in keywords_lower:
                    if keyword in pattern:
                        score += 10
            
            # Check formula patterns
            for formula_pattern in fert_data['formula_patterns']:
                for keyword in keywords_lower:
                    if keyword.upper() in formula_pattern:
                        score += 5
            
            if score > 0:
                composition = fert_data['composition']
                total_content = sum(composition['cations'].values()) + sum(composition['anions'].values())
                
                matching_fertilizers.append({
                    'name': fert_data['patterns'][0].title(),
                    'formula': composition['formula'],
                    'total_content': total_content,
                    'match_score': score,
                    'molecular_weight': composition['mw']
                })
        
        # Sort by match score and total content
        matching_fertilizers.sort(key=lambda x: (x['match_score'], x['total_content']), reverse=True)
        
        return matching_fertilizers

    def get_element_sources_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all available sources for each element
        """
        elements = ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo', 'Cl', 'Na', 'NH4', 'HCO3']
        element_sources = {}
        
        for element in elements:
            sources = self.find_fertilizers_containing_element(element, min_content=0.5)
            
            if sources:
                best_source = sources[0]
                element_sources[element] = {
                    'available_sources': len(sources),
                    'best_source': best_source['name'],
                    'best_content': best_source['element_content'],
                    'all_sources': [s['name'] for s in sources[:5]]  # Top 5
                }
            else:
                element_sources[element] = {
                    'available_sources': 0,
                    'best_source': 'Not available',
                    'best_content': 0,
                    'all_sources': []
                }
        
        return element_sources

    def export_database_summary(self) -> str:
        """
        Export a human-readable summary of the database
        """
        summary_lines = []
        summary_lines.append("FERTILIZER DATABASE SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Total fertilizers: {len(self.fertilizer_data)}")
        summary_lines.append("")
        
        # Group by type
        type_groups = {
            'Acids': [],
            'Nitrates': [],
            'Sulfates': [],
            'Phosphates': [],
            'Chlorides': [],
            'Micronutrients': []
        }
        
        for fert_key, fert_data in self.fertilizer_data.items():
            name = fert_data['patterns'][0].title()
            formula = fert_data['composition']['formula']
            
            if 'acido' in fert_key:
                type_groups['Acids'].append(f"{name} ({formula})")
            elif 'nitrato' in fert_key:
                type_groups['Nitrates'].append(f"{name} ({formula})")
            elif 'sulfato' in fert_key:
                if any(micro in fert_key for micro in ['hierro', 'manganeso', 'zinc', 'cobre']):
                    type_groups['Micronutrients'].append(f"{name} ({formula})")
                else:
                    type_groups['Sulfates'].append(f"{name} ({formula})")
            elif 'fosfato' in fert_key:
                type_groups['Phosphates'].append(f"{name} ({formula})")
            elif 'cloruro' in fert_key:
                type_groups['Chlorides'].append(f"{name} ({formula})")
            else:
                type_groups['Micronutrients'].append(f"{name} ({formula})")
        
        for group_name, fertilizers in type_groups.items():
            if fertilizers:
                summary_lines.append(f"{group_name} ({len(fertilizers)}):")
                for fert in fertilizers:
                    summary_lines.append(f"  - {fert}")
                summary_lines.append("")
        
        # Element coverage
        element_sources = self.get_element_sources_summary()
        summary_lines.append("ELEMENT COVERAGE:")
        summary_lines.append("-" * 20)
        
        for element, info in element_sources.items():
            if info['available_sources'] > 0:
                summary_lines.append(f"{element}: {info['available_sources']} sources (best: {info['best_source']} - {info['best_content']:.1f}%)")
            else:
                summary_lines.append(f"{element}: No sources available")
        
        return "\n".join(summary_lines)

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics
        """
        stats = {
            'total_fertilizers': len(self.fertilizer_data),
            'total_patterns': 0,
            'total_formula_patterns': 0,
            'average_content': 0,
            'content_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'molecular_weight_range': {'min': float('inf'), 'max': 0, 'average': 0},
            'element_availability': {}
        }
        
        total_content_sum = 0
        mw_sum = 0
        
        # Element availability counter
        elements = ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        element_counts = {elem: 0 for elem in elements}
        
        for fert_key, fert_data in self.fertilizer_data.items():
            composition = fert_data['composition']
            
            # Count patterns
            stats['total_patterns'] += len(fert_data['patterns'])
            stats['total_formula_patterns'] += len(fert_data['formula_patterns'])
            
            # Calculate total content
            total_content = sum(composition['cations'].values()) + sum(composition['anions'].values())
            total_content_sum += total_content
            
            # Content distribution
            if total_content >= 50:
                stats['content_distribution']['high'] += 1
            elif total_content >= 20:
                stats['content_distribution']['medium'] += 1
            else:
                stats['content_distribution']['low'] += 1
            
            # Molecular weight statistics
            mw = composition['mw']
            mw_sum += mw
            stats['molecular_weight_range']['min'] = min(stats['molecular_weight_range']['min'], mw)
            stats['molecular_weight_range']['max'] = max(stats['molecular_weight_range']['max'], mw)
            
            # Element availability
            for element in elements:
                cation_content = composition['cations'].get(element, 0)
                anion_content = composition['anions'].get(element, 0)
                if (cation_content + anion_content) > 1:
                    element_counts[element] += 1
        
        # Calculate averages
        if len(self.fertilizer_data) > 0:
            stats['average_content'] = total_content_sum / len(self.fertilizer_data)
            stats['molecular_weight_range']['average'] = mw_sum / len(self.fertilizer_data)
        
        stats['element_availability'] = element_counts
        
        return stats


# Helper functions for standalone usage
def test_fertilizer_database():
    """
    Test function to verify database functionality
    """
    print("[TEST] Testing Fertilizer Database...")
    
    db = FertilizerDatabase()
    
    # Test 1: Database info
    print("\n1. Database Overview:")
    db_info = db.get_complete_database_info()
    print(f"   Total fertilizers: {db_info['total_fertilizers']}")
    print(f"   Validation errors: {len(db_info['validation_report']['validation_errors'])}")
    
    # Test 2: Search functionality
    print("\n2. Search Tests:")
    test_searches = [
        ("nitrato de calcio", "Ca(NO3)2"),
        ("potassium nitrate", "KNO3"),
        ("fosfato monopotasico", "KH2PO4"),
        ("unknown fertilizer", "XYZ123")
    ]
    
    for name, formula in test_searches:
        result = db.find_fertilizer_composition(name, formula)
        if result:
            total_content = sum(result['cations'].values()) + sum(result['anions'].values())
            print(f"   [SUCCESS] {name}: Found ({result['formula']}, {total_content:.1f}% content)")
        else:
            print(f"   [FAILED] {name}: Not found")
    
    # Test 3: Element sources
    print("\n3. Element Sources:")
    for element in ['N', 'P', 'K', 'Ca', 'Fe']:
        sources = db.find_fertilizers_containing_element(element, min_content=5.0)
        if sources:
            best = sources[0]
            print(f"   {element}: {len(sources)} sources (best: {best['name']} - {best['element_content']:.1f}%)")
        else:
            print(f"   {element}: No sources found")
    
    # Test 4: Database integrity
    print("\n4. Database Integrity:")
    validation = db.validate_database_integrity()
    print(f"   Integrity score: {validation['integrity_score']}/100")
    print(f"   Errors: {len(validation['validation_errors'])}")
    print(f"   Warnings: {len(validation['warnings'])}")
    
    print("\n[SUCCESS] Database test completed!")
    return True


if __name__ == "__main__":
    # Run test if executed directly
    test_fertilizer_database()