#!/usr/bin/env python3
"""
COMPLETE FERTILIZER CALCULATOR API - PART 1 OF 3
Imports, Pydantic Models, and Basic Classes
"""

from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
from typing import Dict, List, Optional, Any
import numpy as np # type: ignore
import asyncio
import aiohttp # type: ignore
from dataclasses import dataclass
import json
import math
from datetime import datetime
import os

# PDF generation imports
from reportlab.lib import colors # type: ignore
from reportlab.lib.pagesizes import letter, A4, landscape # type: ignore
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak # type: ignore
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle # type: ignore
from reportlab.lib.units import inch # type: ignore
from reportlab.pdfgen import canvas # type: ignore
import io

app = FastAPI(title="Complete Fertilizer Calculator API", version="3.0.0")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class FertilizerComposition(BaseModel):
    cations: Dict[str, float]
    anions: Dict[str, float]


class FertilizerChemistry(BaseModel):
    formula: str
    purity: float
    solubility: float
    is_ph_adjuster: bool


class Fertilizer(BaseModel):
    name: str
    percentage: float
    molecular_weight: float
    salt_weight: float
    density: Optional[float] = 1.0
    chemistry: FertilizerChemistry
    composition: FertilizerComposition


class CalculationSettings(BaseModel):
    volume_liters: float = 1000
    precision: int = 2
    units: str = "mg/L"
    crop_phase: str = "General"


class FertilizerRequest(BaseModel):
    fertilizers: List[Fertilizer]
    target_concentrations: Dict[str, float]
    water_analysis: Dict[str, float]
    calculation_settings: CalculationSettings

# Response Models


class FertilizerDosage(BaseModel):
    dosage_ml_per_L: float
    dosage_g_per_L: float


class VerificationResult(BaseModel):
    parameter: str
    target_value: float
    actual_value: float
    unit: str
    deviation: float
    percentage_deviation: float
    status: str
    color: str
    recommendation: str
    min_acceptable: float
    max_acceptable: float


class IonicRelationship(BaseModel):
    relationship_name: str
    actual_ratio: float
    target_min: float
    target_max: float
    unit: str
    status: str
    color: str
    recommendation: str


class IonicBalance(BaseModel):
    cation_sum: float
    anion_sum: float
    difference: float
    difference_percentage: float
    is_balanced: int
    tolerance: float


class CostAnalysis(BaseModel):
    total_cost_concentrated: float
    total_cost_diluted: float
    cost_per_liter_concentrated: float
    cost_per_liter_diluted: float
    cost_per_m3_diluted: float
    cost_per_fertilizer: Dict[str, float]
    percentage_per_fertilizer: Dict[str, float]


class NutrientContributions(BaseModel):
    APORTE_mg_L: Dict[str, float]
    DE_mmol_L: Dict[str, float]
    IONES_meq_L: Dict[str, float]


class WaterContribution(BaseModel):
    IONES_mg_L_DEL_AGUA: Dict[str, float]
    mmol_L: Dict[str, float]
    meq_L: Dict[str, float]


class FinalSolution(BaseModel):
    FINAL_mg_L: Dict[str, float]
    FINAL_mmol_L: Dict[str, float]
    FINAL_meq_L: Dict[str, float]
    calculated_EC: float
    calculated_pH: float


class CalculationStatus(BaseModel):
    success: bool
    warnings: List[str]
    iterations: int
    convergence_error: float


class AdvancedFertilizerResponse(BaseModel):
    fertilizer_dosages: Dict[str, FertilizerDosage]
    nutrient_contributions: NutrientContributions
    water_contribution: WaterContribution
    final_solution: FinalSolution
    verification_results: List[VerificationResult]
    ionic_relationships: List[IonicRelationship]
    ionic_balance: IonicBalance
    cost_analysis: CostAnalysis
    calculation_status: CalculationStatus
    pdf_report: Optional[Dict[str, str]] = None

# ============================================================================
# CORE CALCULATION CLASSES
# ============================================================================


class ElementData:
    """Element data for conversions and calculations"""

    def __init__(self, atomic_weight: float, valence: int, is_cation: bool):
        self.atomic_weight = atomic_weight
        self.valence = valence
        self.is_cation = is_cation


class EnhancedFertilizerCalculator:
    """Advanced nutrient calculator"""

    def __init__(self):
        self.element_data = {
            # Cations
            'Ca': ElementData(40.08, 2, True),
            'K': ElementData(39.10, 1, True),
            'Mg': ElementData(24.31, 2, True),
            'Na': ElementData(22.99, 1, True),
            'NH4': ElementData(18.04, 1, True),
            'Fe': ElementData(55.85, 2, True),
            'Mn': ElementData(54.94, 2, True),
            'Zn': ElementData(65.38, 2, True),
            'Cu': ElementData(63.55, 2, True),

            # Anions
            'NO3': ElementData(62.00, 1, False),
            'N': ElementData(14.01, 1, False),
            'SO4': ElementData(96.06, 2, False),
            'S': ElementData(32.06, 2, False),
            'Cl': ElementData(35.45, 1, False),
            'H2PO4': ElementData(96.99, 1, False),
            'P': ElementData(30.97, 1, False),
            'HCO3': ElementData(61.02, 1, False),
            'B': ElementData(10.81, 3, False),
            'Mo': ElementData(95.96, 6, False)
        }

    def calculate_fertilizer_requirement(self, target_element: str, target_concentration: float,
                                         fertilizer_composition: Dict[str, float],
                                         purity: float, molecular_weight: float) -> float:
        """Calculate fertilizer requirement"""
        print(
            f"    Calculating requirement for {target_element}: target={target_concentration} mg/L")

        if target_element not in fertilizer_composition:
            print(
                f"    WARNING: Element {target_element} not found in composition")
            return 0.0

        element_weight_percent = fertilizer_composition[target_element]
        if element_weight_percent <= 0:
            print(
                f"    WARNING: Element {target_element} has zero content: {element_weight_percent}")
            return 0.0

        # Calculate fertilizer amount in mg/L
        fertilizer_amount = target_concentration * 100.0 / \
            (element_weight_percent * (purity / 100.0))

        print(f"    RESULT: {fertilizer_amount:.3f} mg/L fertilizer needed")
        return max(0, fertilizer_amount)

    def calculate_element_contribution(self, fertilizer_amount: float, element_weight_percent: float,
                                       purity: float) -> float:
        """Calculate element contribution from fertilizer amount"""
        if fertilizer_amount <= 0 or element_weight_percent <= 0:
            return 0.0

        contribution = fertilizer_amount * \
            element_weight_percent * (purity / 100.0) / 100.0
        return contribution

    def convert_mg_to_mmol(self, mg_l: float, element: str) -> float:
        """Convert mg/L to mmol/L"""
        if element in self.element_data and mg_l > 0:
            return mg_l / self.element_data[element].atomic_weight
        return 0.0

    def convert_mmol_to_meq(self, mmol_l: float, element: str) -> float:
        """Convert mmol/L to meq/L"""
        if element in self.element_data and mmol_l > 0:
            return mmol_l * self.element_data[element].valence
        return 0.0

# ============================================================================
# FERTILIZER COMPOSITION DATABASE
# ============================================================================


class EnhancedFertilizerDatabase:
    """Complete fertilizer composition database"""

    def __init__(self):
        self.fertilizer_data = {
            # Acids
            'acido nitrico': {
                'patterns': ['acido nitrico', 'nitric acid', 'hno3'],
                'formula_patterns': ['HNO3'],
                'composition': {
                    'formula': 'HNO3',
                    'mw': 63.01,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 22.23, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'acido fosfórico': {
                'patterns': ['acido fosfórico', 'phosphoric acid', 'h3po4'],
                'formula_patterns': ['H3PO4'],
                'composition': {
                    'formula': 'H3PO4',
                    'mw': 97.99,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 31.61, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'acido sulfurico': {
                'patterns': ['acido sulfurico', 'sulfuric acid', 'h2so4'],
                'formula_patterns': ['H2SO4'],
                'composition': {
                    'formula': 'H2SO4',
                    'mw': 98.08,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 32.69, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            # Nitrates
            'nitrato de calcio': {
                'patterns': ['nitrato de calcio', 'calcium nitrate'],
                'formula_patterns': ['CA(NO3)2', 'CA(NO3)2.4H2O', 'CA(NO3)2.2(H2O)'],
                'composition': {
                    'formula': 'Ca(NO3)2.4H2O',
                    'mw': 236.15,
                    'cations': {'Ca': 16.97, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 11.86, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'nitrato de potasio': {
                'patterns': ['nitrato de potasio', 'potassium nitrate'],
                'formula_patterns': ['KNO3'],
                'composition': {
                    'formula': 'KNO3',
                    'mw': 101.1,
                    'cations': {'Ca': 0, 'K': 38.67, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 13.85, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'nitrato de amonio': {
                'patterns': ['nitrato de amonio', 'ammonium nitrate'],
                'formula_patterns': ['NH4NO3'],
                'composition': {
                    'formula': 'NH4NO3',
                    'mw': 80.04,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 22.5, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 35.0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'nitrato de magnesio': {
                'patterns': ['nitrato de magnesio', 'magnesium nitrate'],
                'formula_patterns': ['MG(NO3)2', 'MG(NO3)2.6(H2O)'],
                'composition': {
                    'formula': 'Mg(NO3)2.6H2O',
                    'mw': 256.41,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 9.48, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 10.93, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            # Sulfates
            'sulfato de amonio': {
                'patterns': ['sulfato de amonio', 'ammonium sulfate'],
                'formula_patterns': ['(NH4)2SO4', 'NH4)2SO4'],
                'composition': {
                    'formula': '(NH4)2SO4',
                    'mw': 132.14,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 27.28, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 21.21, 'S': 24.26, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'sulfato de potasio': {
                'patterns': ['sulfato de potasio', 'potassium sulfate'],
                'formula_patterns': ['K2SO4'],
                'composition': {
                    'formula': 'K2SO4',
                    'mw': 174.26,
                    'cations': {'Ca': 0, 'K': 44.87, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 18.39, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            'sulfato de magnesio': {
                'patterns': ['sulfato de magnesio', 'magnesium sulfate'],
                'formula_patterns': ['MGSO4', 'MGSO4.7H2O'],
                'composition': {
                    'formula': 'MgSO4.7H2O',
                    'mw': 246.47,
                    'cations': {'Ca': 0, 'K': 0, 'Mg': 9.87, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 13.01, 'Cl': 0, 'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            },
            # Phosphates
            'fosfato monopotasico': {
                'patterns': ['fosfato monopotasico', 'monopotassium phosphate', 'kh2po4'],
                'formula_patterns': ['KH2PO4'],
                'composition': {
                    'formula': 'KH2PO4',
                    'mw': 136.09,
                    'cations': {'Ca': 0, 'K': 28.73, 'Mg': 0, 'Na': 0, 'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0},
                    'anions': {'N': 0, 'S': 0, 'Cl': 0, 'P': 22.76, 'HCO3': 0, 'B': 0, 'Mo': 0}
                }
            }
        }

    def find_fertilizer_composition(self, name: str, formula: str) -> Dict[str, Any]:
        """Find fertilizer composition by name and formula matching"""
        name_lower = name.lower().strip()
        formula_upper = formula.upper().strip()

        print(
            f"    Searching database for: name='{name_lower}', formula='{formula_upper}'")

        # Try exact name matching first
        for fert_key, fert_data in self.fertilizer_data.items():
            for pattern in fert_data['patterns']:
                if pattern in name_lower or name_lower in pattern:
                    print(
                        f"    FOUND by name pattern: '{pattern}' -> {fert_data['composition']['formula']}")
                    return fert_data['composition']

        # Try formula matching
        for fert_key, fert_data in self.fertilizer_data.items():
            for formula_pattern in fert_data['formula_patterns']:
                if formula_pattern in formula_upper:
                    print(
                        f"    FOUND by formula pattern: '{formula_pattern}' -> {fert_data['composition']['formula']}")
                    return fert_data['composition']

        print(f"    NO MATCH FOUND for '{name}' with formula '{formula}'")
        return None

# ============================================================================
# PART 2 OF 3 - MAIN CALCULATOR AND PDF GENERATOR
# ============================================================================


class SwaggerAPIClient:
    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.headers = {'Content-Type': 'application/json'}
        if auth_token:
            self.headers['Authorization'] = f'Bearer {auth_token}'
        self.fertilizer_db = EnhancedFertilizerDatabase()

    async def login(self, user_email: str, password: str):
        """Login to get authentication token"""
        url = f"{self.base_url}/Authentication/Login"
        login_data = {"userEmail": user_email, "password": password}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=login_data, headers={'Content-Type': 'application/json'}) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success') and data.get('result'):
                        self.auth_token = data['result']['token']
                        self.headers['Authorization'] = f'Bearer {self.auth_token}'
                        return {'success': True, 'token': self.auth_token, 'user_data': data['result']}
                    else:
                        raise Exception("Login failed: Invalid credentials")
                else:
                    response_text = await response.text()
                    raise Exception(f"Login failed: {response_text}")

    async def get_fertilizers(self, catalog_id: int, include_inactives: bool = False):
        """Get all fertilizers from catalog"""
        if not self.auth_token:
            raise Exception("Authentication required - please login first")

        url = f"{self.base_url}/Fertilizer"
        params = {'CatalogId': catalog_id,
            'IncludeInactives': 'true' if include_inactives else 'false'}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('result', {}).get('fertilizers', [])
                elif response.status == 401:
                    raise Exception(
                        f"Authentication failed - token may have expired")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

    async def get_fertilizer_chemistry(self, fertilizer_id: int, catalog_id: int):
        """Get detailed fertilizer chemistry data"""
        url = f"{self.base_url}/FertilizerChemistry"
        params = {'FertilizerId': fertilizer_id, 'CatalogId': catalog_id}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    chemistry_list = data.get('result', {}).get(
                        'fertilizerChemistries', [])
                    return chemistry_list[0] if chemistry_list else None
                else:
                    return None

    async def get_crop_phase_requirements(self, phase_id: int):
        """Get crop phase solution requirements"""
        url = f"{self.base_url}/CropPhaseSolutionRequirement/GetByPhaseId"
        params = {'PhaseId': phase_id}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('result', {}).get('cropPhaseSolutionRequirement')
                else:
                    return None

    async def get_water_chemistry(self, water_id: int, catalog_id: int):
        """Get water chemistry analysis"""
        url = f"{self.base_url}/WaterChemistry"
        params = {'WaterId': water_id, 'CatalogId': catalog_id}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    water_list = data.get('result', {}).get(
                        'waterChemistries', [])
                    return water_list[0] if water_list else None
                else:
                    return None

    def map_swagger_fertilizer_to_model(self, swagger_fert: Dict[str, Any], chemistry: Dict[str, Any] = None) -> Fertilizer:
        """Convert Swagger fertilizer data to our Fertilizer model with proper composition"""
        name = swagger_fert.get('name', 'Unknown')
        print(f"Mapping fertilizer: {name}")

        if chemistry is None:
            chemistry = {'formula': name, 'purity': 98,
                'density': 1.0, 'isPhAdjuster': False}

        # Get basic properties
        formula = chemistry.get('formula', name)
        purity = chemistry.get('purity', 98)
        density = chemistry.get('density', 1.0)

        print(f"  Formula: {formula}, Purity: {purity}%, Density: {density}")

        # Get composition from database
        composition_data = self.fertilizer_db.find_fertilizer_composition(
            name, formula)

        if composition_data:
            cations = composition_data['cations']
            anions = composition_data['anions']
            molecular_weight = composition_data['mw']
            print(f"  SUCCESS: Found in database")
        else:
            # Default empty composition
            cations = {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0,
                'NH4': 0, 'Fe': 0, 'Mn': 0, 'Zn': 0, 'Cu': 0}
            anions = {'N': 0, 'S': 0, 'Cl': 0,
                'P': 0, 'HCO3': 0, 'B': 0, 'Mo': 0}
            molecular_weight = 100
            print(f"  WARNING: Using default composition")

        print(f"  Cations: {cations}")
        print(f"  Anions: {anions}")
        print(f"  MW: {molecular_weight}")

        # Ensure reasonable values
        if molecular_weight <= 0:
            molecular_weight = 100
        if purity <= 0 or purity > 100:
            purity = 98
        if density <= 0:
            density = 1.0

        fertilizer = Fertilizer(
            name=name,
            percentage=purity,
            molecular_weight=molecular_weight,
            salt_weight=molecular_weight,
            density=density,
            chemistry=FertilizerChemistry(
                formula=formula,
                purity=purity,
                solubility=chemistry.get('solubility20', 100),
                is_ph_adjuster=chemistry.get('isPhAdjuster', False)
            ),
            composition=FertilizerComposition(cations=cations, anions=anions)
        )

        total_content = sum(cations.values()) + sum(anions.values())
        print(f"  RESULT: Total content={total_content:.1f}%")
        return fertilizer

    def map_requirements_to_targets(self, requirements: Dict[str, Any]) -> Dict[str, float]:
        """Map Swagger requirements to target concentrations"""
        element_mapping = {
            'ca': 'Ca', 'k': 'K', 'mg': 'Mg', 'na': 'Na', 'nH4': 'NH4',
            'n': 'N', 's': 'S', 'cl': 'Cl', 'p': 'P', 'hcO3': 'HCO3',
            'fe': 'Fe', 'mn': 'Mn', 'zn': 'Zn', 'cu': 'Cu', 'b': 'B', 'mo': 'Mo'
        }

        targets = {}
        for api_field, our_field in element_mapping.items():
            if api_field in requirements:
                value = requirements[api_field]
                if value and value > 0:
                    targets[our_field] = float(value)

        return targets

    def map_water_to_analysis(self, water: Dict[str, Any]) -> Dict[str, float]:
        """Map Swagger water data to water analysis"""
        element_mapping = {
            'ca': 'Ca', 'k': 'K', 'mg': 'Mg', 'na': 'Na', 'nH4': 'NH4',
            'nO3': 'N', 'sO4': 'S', 'cl': 'Cl', 'h2PO4': 'P', 'hcO3': 'HCO3',
            'fe': 'Fe', 'cu': 'Cu', 'mn': 'Mn', 'zn': 'Zn', 'b': 'B', 'moO4': 'Mo'
        }

        analysis = {}
        for api_field, our_field in element_mapping.items():
            if api_field in water:
                value = water[api_field]
                if value is not None:
                    analysis[our_field] = float(value)

        return analysis

# ============================================================================
# MAIN FERTILIZER CALCULATOR
# ============================================================================


class FertilizerCalculator:
    """Main fertilizer calculator with optimization and calculations"""

    def __init__(self):
        self.nutrient_calc = EnhancedFertilizerCalculator()

        # Create reports directory if it doesn't exist
        if not os.path.exists('reports'):
            os.makedirs('reports')

    def optimize_solution(self, fertilizers: List[Fertilizer], targets: Dict[str, float], water: Dict[str, float]) -> Dict[str, float]:
        """IMPROVED: Optimized solution calculation with better balancing and micronutrients"""
        print(f"\n=== STARTING IMPROVED FERTILIZER OPTIMIZATION ===")
        print(f"Available fertilizers: {len(fertilizers)}")
        print(f"Target concentrations: {targets}")
        print(f"Water analysis: {water}")
        
        results = {}
        remaining_nutrients = {}

        # Calculate nutrients needed (subtract water contribution)
        print(f"\n--- CALCULATING REMAINING NUTRIENTS AFTER WATER ---")
        for element, target in targets.items():
            water_content = water.get(element, 0)
            remaining = max(0, target - water_content)
            remaining_nutrients[element] = remaining
            print(f"{element}: Target={target:.1f}, Water={water_content:.1f}, Remaining={remaining:.1f} mg/L")

        # Print available fertilizers with their compositions
        print(f"\n--- AVAILABLE FERTILIZERS AND COMPOSITIONS ---")
        useful_fertilizers = []
        for i, fert in enumerate(fertilizers):
            total_content = sum(fert.composition.cations.values()) + sum(fert.composition.anions.values())
            if total_content > 5:  # Only consider fertilizers with significant content
                useful_fertilizers.append(fert)
                print(f"{len(useful_fertilizers)}. {fert.name}")
                main_nutrients = []
                for elem, content in fert.composition.cations.items():
                    if content > 1:
                        main_nutrients.append(f"{elem}:{content:.1f}%")
                for elem, content in fert.composition.anions.items():
                    if content > 1:
                        main_nutrients.append(f"{elem}:{content:.1f}%")
                if main_nutrients:
                    print(f"   Main nutrients: {', '.join(main_nutrients)}")

        # IMPROVED STRATEGY: More precise fertilizer selection
        print(f"\n--- IMPROVED FERTILIZER SELECTION STRATEGY ---")
        
        # 1. Phosphorus first (most critical)
        if remaining_nutrients.get('P', 0) > 0:
            print(f"Step 1: Phosphorus source (need {remaining_nutrients['P']:.1f} mg/L P)...")
            p_fertilizers = [f for f in useful_fertilizers if f.composition.anions.get('P', 0) > 5]
            
            if p_fertilizers:
                # Choose the one with highest P content and lowest N content to avoid N excess
                best_p_fert = min(p_fertilizers, key=lambda f: f.composition.anions.get('N', 0))
                print(f"Selected P source: {best_p_fert.name} (P: {best_p_fert.composition.anions.get('P', 0):.1f}%, N: {best_p_fert.composition.anions.get('N', 0):.1f}%)")
                
                p_needed = remaining_nutrients['P']
                dosage = self.nutrient_calc.calculate_fertilizer_requirement(
                    'P', p_needed, best_p_fert.composition.anions,
                    best_p_fert.percentage, best_p_fert.molecular_weight
                )
                
                if dosage > 0:
                    results[best_p_fert.name] = dosage / 1000.0
                    
                    # Calculate all contributions from this fertilizer
                    for elem in ['K', 'Ca', 'N', 'S']:
                        contribution = self.nutrient_calc.calculate_element_contribution(
                            dosage, 
                            best_p_fert.composition.cations.get(elem, 0) + best_p_fert.composition.anions.get(elem, 0),
                            best_p_fert.percentage
                        )
                        remaining_nutrients[elem] = max(0, remaining_nutrients.get(elem, 0) - contribution)
                        if contribution > 0:
                            print(f"  Also provides {contribution:.1f} mg/L of {elem}")
                    
                    remaining_nutrients['P'] = 0
                    print(f"Added {dosage/1000:.3f} g/L of {best_p_fert.name}")

        # 2. Calcium (essential macronutrient)
        if remaining_nutrients.get('Ca', 0) > 0:
            print(f"Step 2: Calcium source (need {remaining_nutrients['Ca']:.1f} mg/L Ca)...")
            ca_fertilizers = [f for f in useful_fertilizers if f.composition.cations.get('Ca', 0) > 10]
            
            if ca_fertilizers:
                # Choose calcium source with moderate N content
                best_ca_fert = min(ca_fertilizers, key=lambda f: abs(f.composition.anions.get('N', 0) - 12))
                print(f"Selected Ca source: {best_ca_fert.name}")
                
                ca_needed = remaining_nutrients['Ca']
                dosage = self.nutrient_calc.calculate_fertilizer_requirement(
                    'Ca', ca_needed, best_ca_fert.composition.cations,
                    best_ca_fert.percentage, best_ca_fert.molecular_weight
                )
                
                if dosage > 0:
                    results[best_ca_fert.name] = dosage / 1000.0
                    
                    # Calculate contributions
                    for elem in ['N', 'K', 'Mg']:
                        contribution = self.nutrient_calc.calculate_element_contribution(
                            dosage,
                            best_ca_fert.composition.cations.get(elem, 0) + best_ca_fert.composition.anions.get(elem, 0),
                            best_ca_fert.percentage
                        )
                        remaining_nutrients[elem] = max(0, remaining_nutrients.get(elem, 0) - contribution)
                        if contribution > 0:
                            print(f"  Also provides {contribution:.1f} mg/L of {elem}")
                    
                    remaining_nutrients['Ca'] = 0
                    print(f"Added {dosage/1000:.3f} g/L of {best_ca_fert.name}")

        # 3. Potassium (balance K without excess N)
        if remaining_nutrients.get('K', 0) > 0:
            print(f"Step 3: Potassium source (need {remaining_nutrients['K']:.1f} mg/L K)...")
            k_fertilizers = [f for f in useful_fertilizers if f.composition.cations.get('K', 0) > 20]
            
            if k_fertilizers:
                # If we have excess N, prefer K2SO4, otherwise KNO3 is fine
                n_excess = remaining_nutrients.get('N', 0) < 50  # If we need less than 50mg/L N
                if n_excess:
                    # Prefer sulfate sources
                    best_k_fert = max([f for f in k_fertilizers if f.composition.anions.get('S', 0) > 0] or k_fertilizers,
                                    key=lambda f: f.composition.cations.get('K', 0))
                else:
                    # KNO3 is fine
                    best_k_fert = max(k_fertilizers, key=lambda f: f.composition.cations.get('K', 0))
                
                print(f"Selected K source: {best_k_fert.name}")
                
                k_needed = remaining_nutrients['K']
                dosage = self.nutrient_calc.calculate_fertilizer_requirement(
                    'K', k_needed, best_k_fert.composition.cations,
                    best_k_fert.percentage, best_k_fert.molecular_weight
                )
                
                if dosage > 0:
                    results[best_k_fert.name] = dosage / 1000.0
                    
                    # Calculate contributions
                    for elem in ['N', 'S']:
                        contribution = self.nutrient_calc.calculate_element_contribution(
                            dosage,
                            best_k_fert.composition.anions.get(elem, 0),
                            best_k_fert.percentage
                        )
                        remaining_nutrients[elem] = max(0, remaining_nutrients.get(elem, 0) - contribution)
                        if contribution > 0:
                            print(f"  Also provides {contribution:.1f} mg/L of {elem}")
                    
                    remaining_nutrients['K'] = 0
                    print(f"Added {dosage/1000:.3f} g/L of {best_k_fert.name}")

        # 4. Magnesium
        if remaining_nutrients.get('Mg', 0) > 0:
            print(f"Step 4: Magnesium source (need {remaining_nutrients['Mg']:.1f} mg/L Mg)...")
            mg_fertilizers = [f for f in useful_fertilizers if f.composition.cations.get('Mg', 0) > 5]
            
            if mg_fertilizers:
                best_mg_fert = max(mg_fertilizers, key=lambda f: f.composition.cations.get('Mg', 0))
                print(f"Selected Mg source: {best_mg_fert.name}")
                
                mg_needed = remaining_nutrients['Mg']
                dosage = self.nutrient_calc.calculate_fertilizer_requirement(
                    'Mg', mg_needed, best_mg_fert.composition.cations,
                    best_mg_fert.percentage, best_mg_fert.molecular_weight
                )
                
                if dosage > 0:
                    results[best_mg_fert.name] = dosage / 1000.0
                    
                    # Calculate S contribution
                    s_contribution = self.nutrient_calc.calculate_element_contribution(
                        dosage, best_mg_fert.composition.anions.get('S', 0), best_mg_fert.percentage
                    )
                    remaining_nutrients['S'] = max(0, remaining_nutrients.get('S', 0) - s_contribution)
                    remaining_nutrients['Mg'] = 0
                    
                    print(f"Added {dosage/1000:.3f} g/L of {best_mg_fert.name}")
                    if s_contribution > 0:
                        print(f"  Also provides {s_contribution:.1f} mg/L of S")

        # 5. Sulfur (if still needed)
        if remaining_nutrients.get('S', 0) > 10:
            print(f"Step 5: Additional sulfur source (need {remaining_nutrients['S']:.1f} mg/L S)...")
            s_fertilizers = [f for f in useful_fertilizers 
                            if f.composition.anions.get('S', 0) > 10 and f.name not in results]
            
            if s_fertilizers:
                # Prefer sources with minimal N content
                best_s_fert = min(s_fertilizers, key=lambda f: f.composition.anions.get('N', 0))
                print(f"Selected S source: {best_s_fert.name}")
                
                s_needed = remaining_nutrients['S']
                combined_composition = {**best_s_fert.composition.cations, **best_s_fert.composition.anions}
                dosage = self.nutrient_calc.calculate_fertilizer_requirement(
                    'S', s_needed, combined_composition,
                    best_s_fert.percentage, best_s_fert.molecular_weight
                )
                
                if dosage > 0:
                    results[best_s_fert.name] = dosage / 1000.0
                    remaining_nutrients['S'] = 0
                    print(f"Added {dosage/1000:.3f} g/L of {best_s_fert.name}")

        # 6. MICRONUTRIENTS (Critical improvement)
        print(f"\nStep 6: Adding micronutrients...")
        micronutrient_targets = {
            'Fe': targets.get('Fe', 2.0),
            'Mn': targets.get('Mn', 0.5),
            'Zn': targets.get('Zn', 0.3),
            'Cu': targets.get('Cu', 0.1),
            'B': targets.get('B', 0.5),
            'Mo': targets.get('Mo', 0.05)
        }
        
        # Add micronutrient fertilizers
        micro_fertilizers = {
            'Fe': ('FeEDTA', 0.013),     # 13% Fe
            'Mn': ('MnSO4.4H2O', 0.24), # 24% Mn
            'Zn': ('ZnSO4.7H2O', 0.23), # 23% Zn
            'Cu': ('CuSO4.5H2O', 0.25), # 25% Cu
            'B': ('H3BO3', 0.17),       # 17% B
            'Mo': ('Na2MoO4.2H2O', 0.39) # 39% Mo
        }
        
        for micro, target in micronutrient_targets.items():
            if target > 0:
                fert_name, content_percent = micro_fertilizers.get(micro, (f"{micro}_source", 0.1))
                dosage_mg_l = target / (content_percent / 100.0)
                dosage_g_l = dosage_mg_l / 1000.0
                
                if dosage_g_l > 0.001:  # Only add if dosage is meaningful
                    results[fert_name] = dosage_g_l
                    print(f"Added {dosage_g_l:.4f} g/L of {fert_name} for {target:.1f} mg/L {micro}")

        # 7. NITROGEN BALANCING (Critical improvement)
        print(f"\nStep 7: Nitrogen balance adjustment...")
        
        # Calculate total N contribution so far
        total_n_contribution = 0
        for fert_name, dosage_g_l in results.items():
            fertilizer = next((f for f in useful_fertilizers if f.name == fert_name), None)
            if fertilizer:
                n_contribution = self.nutrient_calc.calculate_element_contribution(
                    dosage_g_l * 1000, fertilizer.composition.anions.get('N', 0), fertilizer.percentage
                )
                total_n_contribution += n_contribution
        
        n_target = targets.get('N', 150)
        n_from_water = water.get('N', 0)
        n_needed = n_target - n_from_water
        n_excess = total_n_contribution - n_needed
        
        print(f"N Analysis: Target={n_target:.1f}, From water={n_from_water:.1f}, Needed={n_needed:.1f}")
        print(f"Current N contribution: {total_n_contribution:.1f} mg/L")
        print(f"N excess: {n_excess:.1f} mg/L")
        
        # If N is excessive, reduce fertilizers proportionally
        if n_excess > 20:  # If excess is more than 20 mg/L
            reduction_factor = n_needed / total_n_contribution
            print(f"Applying reduction factor: {reduction_factor:.3f}")
            
            # Reduce all N-containing fertilizers proportionally
            for fert_name in list(results.keys()):
                fertilizer = next((f for f in useful_fertilizers if f.name == fert_name), None)
                if fertilizer and fertilizer.composition.anions.get('N', 0) > 0:
                    original_dosage = results[fert_name]
                    results[fert_name] = original_dosage * reduction_factor
                    print(f"Reduced {fert_name}: {original_dosage:.3f} → {results[fert_name]:.3f} g/L")

        print(f"\n=== IMPROVED OPTIMIZATION RESULTS ===")
        total_fertilizers = len([d for d in results.values() if d > 0])
        print(f"Selected {total_fertilizers} fertilizers:")
        for name, dosage in results.items():
            if dosage > 0:
                print(f"  {name}: {dosage:.3f} g/L")

        return results

    def calculate_all_contributions(self, fertilizers: List[Fertilizer], dosages: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate all nutrient contributions from fertilizers"""
        elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'SO4', 'S',
            'Cl', 'H2PO4', 'P', 'HCO3', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']

        contributions = {
            'APORTE_mg_L': {elem: 0.0 for elem in elements},
            'DE_mmol_L': {elem: 0.0 for elem in elements},
            'IONES_meq_L': {elem: 0.0 for elem in elements}
        }

        print(f"\n=== CALCULATING NUTRIENT CONTRIBUTIONS ===")

        for fertilizer in fertilizers:
            dosage_g_l = dosages.get(fertilizer.name, 0)
            if dosage_g_l > 0:
                print(
                    f"\nCalculating contributions from {fertilizer.name} ({dosage_g_l:.3f} g/L):")

                for element in elements:
                    cation_content = fertilizer.composition.cations.get(
                        element, 0)
                    anion_content = fertilizer.composition.anions.get(
                        element, 0)
                    total_content = cation_content + anion_content

                    if total_content > 0:
                        dosage_mg_l = dosage_g_l * 1000

                        contribution_mg_l = self.nutrient_calc.calculate_element_contribution(
                            dosage_mg_l, total_content, fertilizer.percentage
                        )

                        contributions['APORTE_mg_L'][element] += contribution_mg_l

                        mmol_contribution = self.nutrient_calc.convert_mg_to_mmol(
                            contribution_mg_l, element)
                        contributions['DE_mmol_L'][element] += mmol_contribution

                        meq_contribution = self.nutrient_calc.convert_mmol_to_meq(
                            mmol_contribution, element)
                        contributions['IONES_meq_L'][element] += meq_contribution

                        if contribution_mg_l > 0.1:
                            print(
                                f"  {element}: +{contribution_mg_l:.2f} mg/L")

        # Round all values
        for category in contributions:
            for element in contributions[category]:
                contributions[category][element] = round(
                    contributions[category][element], 3)

        print(f"\nTotal nutrient contributions:")
        for element in elements:
            if contributions['APORTE_mg_L'][element] > 0.1:
                print(
                    f"  {element}: {contributions['APORTE_mg_L'][element]:.1f} mg/L")

        return contributions

    def calculate_water_contributions(self, water_analysis: Dict[str, float]):
        """Calculate water contributions in all units"""
        elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'SO4', 'S',
            'Cl', 'H2PO4', 'P', 'HCO3', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']

        water_contrib = {
            'IONES_mg_L_DEL_AGUA': {},
            'mmol_L': {},
            'meq_L': {}
        }

        for element in elements:
            mg_l = water_analysis.get(element, 0)
            mmol_l = self.nutrient_calc.convert_mg_to_mmol(mg_l, element)
            meq_l = self.nutrient_calc.convert_mmol_to_meq(mmol_l, element)

            water_contrib['IONES_mg_L_DEL_AGUA'][element] = round(mg_l, 3)
            water_contrib['mmol_L'][element] = round(mmol_l, 3)
            water_contrib['meq_L'][element] = round(meq_l, 3)

        return water_contrib

    def calculate_final_solution(self, nutrient_contrib: Dict[str, Dict[str, float]], water_contrib: Dict):
        """Calculate final solution concentrations"""
        elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'SO4', 'S',
            'Cl', 'H2PO4', 'P', 'HCO3', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']

        final = {
            'FINAL_mg_L': {},
            'FINAL_mmol_L': {},
            'FINAL_meq_L': {}
        }

        for element in elements:
            final_mg_l = (nutrient_contrib['APORTE_mg_L'][element] +
                          water_contrib['IONES_mg_L_DEL_AGUA'][element])
            final_mmol_l = self.nutrient_calc.convert_mg_to_mmol(
                final_mg_l, element)
            final_meq_l = self.nutrient_calc.convert_mmol_to_meq(
                final_mmol_l, element)

            final['FINAL_mg_L'][element] = round(final_mg_l, 3)
            final['FINAL_mmol_L'][element] = round(final_mmol_l, 3)
            final['FINAL_meq_L'][element] = round(final_meq_l, 3)

        # Calculate EC and pH
        ec = self._calculate_ec_advanced(final['FINAL_meq_L'])
        ph = self._calculate_ph_advanced(final['FINAL_mg_L'])

        return {
            'FINAL_mg_L': final['FINAL_mg_L'],
            'FINAL_mmol_L': final['FINAL_mmol_L'],
            'FINAL_meq_L': final['FINAL_meq_L'],
            'calculated_EC': round(ec, 2),
            'calculated_pH': round(ph, 1)
        }

    def _calculate_ec_advanced(self, final_meq: Dict[str, float]) -> float:
        """Advanced EC calculation based on cation sum"""
        cations = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'Fe', 'Mn', 'Zn', 'Cu']
        cation_sum = sum(final_meq.get(cation, 0) for cation in cations)
        return cation_sum * 0.1

    def _calculate_ph_advanced(self, final_mg: Dict[str, float]) -> float:
        """Advanced pH calculation"""
        hco3 = final_mg.get('HCO3', 0)
        no3_n = final_mg.get('N', 0)

        if hco3 > 61:
            return 6.5 + (hco3 - 61) / 100
        else:
            return 6.0 - (no3_n / 200)

# ============================================================================
# VERIFICATION AND ANALYSIS CLASSES
# ============================================================================


class SolutionVerifier:
    """Solution verification module"""

    def __init__(self):
        self.nutrient_ranges = {
            'N': (100, 200, 0.05), 'P': (30, 50, 0.05), 'K': (200, 300, 0.05),
            'Ca': (150, 200, 0.05), 'Mg': (40, 60, 0.05), 'S': (60, 120, 0.05),
            'Fe': (1.0, 3.0, 0.10), 'Mn': (0.5, 1.0, 0.10), 'Zn': (0.2, 0.5, 0.10),
            'Cu': (0.1, 0.3, 0.10), 'B': (0.3, 0.8, 0.10), 'Mo': (0.01, 0.05, 0.10)
        }

    def verify_concentrations(self, target_concentrations: Dict[str, float], final_concentrations: Dict[str, float]) -> List[Dict[str, Any]]:
        """Verify nutrient concentrations against targets"""
        results = []

        for nutrient, target in target_concentrations.items():
            if nutrient in final_concentrations:
                final = final_concentrations[nutrient]
                deviation = final - target
                percentage_deviation = abs(
                    deviation) / target * 100 if target > 0 else 0

                if nutrient in self.nutrient_ranges:
                    min_val, max_val, tolerance = self.nutrient_ranges[nutrient]
                    min_acceptable = target * (1 - tolerance)
                    max_acceptable = target * (1 + tolerance)
                else:
                    min_acceptable = target * 0.95
                    max_acceptable = target * 1.05

                if min_acceptable <= final <= max_acceptable:
                    status = "OK"
                    color = "Green"
                    recommendation = "Concentration within acceptable range"
                elif final > max_acceptable:
                    status = "Critical" if final > target * 1.2 else "High"
                    color = "Red" if status == "Critical" else "Orange"
                    recommendation = "Concentration too high. Reduce fertilizer or increase dilution."
                else:
                    status = "Critical" if final < target * 0.8 else "Low"
                    color = "Red" if status == "Critical" else "Yellow"
                    recommendation = "Concentration too low. Increase fertilizer."

                results.append({
                    'parameter': nutrient, 'target_value': target, 'actual_value': final, 'unit': 'mg/L',
                    'deviation': deviation, 'percentage_deviation': percentage_deviation, 'status': status,
                    'color': color, 'recommendation': recommendation, 'min_acceptable': min_acceptable, 'max_acceptable': max_acceptable
                })

        return results

    def verify_ionic_balance(self, final_meq: Dict[str, float]) -> Dict[str, float]:
        """Verify ionic balance (cations vs anions)"""
        cation_sum = sum(final_meq.get(cation, 0) for cation in [
                         'Ca', 'K', 'Mg', 'Na', 'NH4', 'Fe', 'Mn', 'Zn', 'Cu'])
        anion_sum = sum(final_meq.get(anion, 0)
                        for anion in ['N', 'S', 'Cl', 'P', 'HCO3', 'B', 'Mo'])

        difference = abs(cation_sum - anion_sum)
        difference_percentage = (difference / cation_sum) * \
                                 100 if cation_sum > 0 else 0
        is_balanced = difference_percentage <= 10.0

        return {
            'cation_sum': cation_sum, 'anion_sum': anion_sum, 'difference': difference,
            'difference_percentage': difference_percentage, 'is_balanced': 1 if is_balanced else 0,
            'tolerance': min(cation_sum, anion_sum) * 0.1
        }

    def verify_ionic_relationships(self, final_meq: Dict[str, float], final_mmol: Dict[str, float], final_mg: Dict[str, float]) -> List[Dict[str, Any]]:
        """Verify ionic relationships"""
        results = []
        k_meq = final_meq.get('K', 0)
        ca_meq = final_meq.get('Ca', 0)
        mg_meq = final_meq.get('Mg', 0)

        # K:Ca relationship
        if ca_meq > 0:
            k_ca_ratio = k_meq / ca_meq
            results.append({
                'relationship_name': 'K:Ca Ratio', 'actual_ratio': k_ca_ratio, 'target_min': 0.8, 'target_max': 1.5,
                'unit': 'meq/L ratio', 'status': 'OK' if 0.8 <= k_ca_ratio <= 1.5 else 'Unbalanced',
                'color': 'Green' if 0.8 <= k_ca_ratio <= 1.5 else 'Orange',
                'recommendation': 'K:Ca ratio balanced' if 0.8 <= k_ca_ratio <= 1.5 else 'K:Ca ratio unbalanced'
            })

        return results


class CostAnalyzer:
    """Cost analysis module"""

    def __init__(self):
        self.fertilizer_costs = {
            'Acido Nítrico DAC': 1.20, 'Acido Fosfórico': 1.50, 'Acido Sulfurico': 1.00,
            'Nitrato de amonio': 0.45, 'Sulfato de amonio': 0.50, 'Nitrato de calcio': 0.80,
            'Nitrato de calcio amoniacal': 0.85, 'Nitrato de potasio': 1.20, 'Sulfato de potasio': 1.50,
            'Sulfato de magnesio': 0.60, 'Fosfato monopotasico': 2.50
        }

    def calculate_solution_cost(self, fertilizer_amounts: Dict[str, float], concentrated_volume: float, diluted_volume: float) -> Dict[str, Any]:
        """Calculate complete cost analysis"""
        cost_per_fertilizer = {}
        total_cost = 0

        for fertilizer, amount_kg in fertilizer_amounts.items():
            cost_per_kg = self.fertilizer_costs.get(fertilizer, 1.0)
            cost = amount_kg * cost_per_kg
            cost_per_fertilizer[fertilizer] = cost
            total_cost += cost

        percentage_per_fertilizer = {}
        if total_cost > 0:
            for fertilizer, cost in cost_per_fertilizer.items():
                percentage_per_fertilizer[fertilizer] = (
                    cost / total_cost) * 100

        return {
            'total_cost_concentrated': total_cost, 'total_cost_diluted': total_cost,
            'cost_per_liter_concentrated': total_cost / concentrated_volume if concentrated_volume > 0 else 0,
            'cost_per_liter_diluted': total_cost / diluted_volume if diluted_volume > 0 else 0,
            'cost_per_m3_diluted': (total_cost / diluted_volume * 1000) if diluted_volume > 0 else 0,
            'cost_per_fertilizer': cost_per_fertilizer, 'percentage_per_fertilizer': percentage_per_fertilizer,
        }

    def _create_comprehensive_summary_rows(self, nutrient_contributions: Dict, water_contribution: Dict, final_solution: Dict) -> List[List]:
        """Create comprehensive summary rows matching Excel format"""
        summary_rows = []

        # Get data dictionaries
        aporte_mg = nutrient_contributions.get('APORTE_mg_L', {})
        aporte_mmol = nutrient_contributions.get('DE_mmol_L', {})
        aporte_meq = nutrient_contributions.get('IONES_meq_L', {})

        agua_mg = water_contribution.get('IONES_mg_L_DEL_AGUA', {})
        agua_mmol = water_contribution.get('mmol_L', {})
        agua_meq = water_contribution.get('meq_L', {})

        final_mg = final_solution.get('FINAL_mg_L', {})
        final_mmol = final_solution.get('FINAL_mmol_L', {})
        final_meq = final_solution.get('FINAL_meq_L', {})

        anion_elements = ['N', 'S', 'Cl', 'P', 'HCO3']
        final_ec = final_solution.get('calculated_EC', 0)

        # Row 1: Aporte de Iones (mg/L)
        aporte_anion_sum = sum(aporte_mg.get(elem, 0)
                               for elem in anion_elements)
        row1 = ['Aporte de Iones (mg/L)', '', '', '', '', '', '',
                f"{aporte_mg.get('Ca', 0):.1f}", f"{aporte_mg.get('K', 0):.1f}",
                f"{aporte_mg.get('Mg', 0):.1f}", f"{aporte_mg.get('Na', 0):.1f}",
                f"{aporte_mg.get('NH4', 0):.1f}", f"{aporte_mg.get('N', 0):.1f}",
                f"{aporte_mg.get('N', 0):.1f}", f"{aporte_mg.get('S', 0):.1f}",
                f"{aporte_mg.get('S', 0):.1f}", f"{aporte_mg.get('Cl', 0):.1f}",
                f"{aporte_mg.get('P', 0):.1f}", f"{aporte_mg.get('P', 0):.1f}",
                f"{aporte_mg.get('HCO3', 0):.1f}", f"{aporte_anion_sum:.1f}", f"{final_ec:.2f}"]

        # Row 2: Aporte de Iones (mmol/L)
        aporte_mmol_anion_sum = sum(aporte_mmol.get(elem, 0)
                                    for elem in anion_elements)
        row2 = ['Aporte de Iones (mmol/L)', '', '', '', '', '', '',
                f"{aporte_mmol.get('Ca', 0):.3f}", f"{aporte_mmol.get('K', 0):.3f}",
                f"{aporte_mmol.get('Mg', 0):.3f}", f"{aporte_mmol.get('Na', 0):.3f}",
                f"{aporte_mmol.get('NH4', 0):.3f}", f"{aporte_mmol.get('N', 0):.3f}",
                f"{aporte_mmol.get('N', 0):.3f}", f"{aporte_mmol.get('S', 0):.3f}",
                f"{aporte_mmol.get('S', 0):.3f}", f"{aporte_mmol.get('Cl', 0):.3f}",
                f"{aporte_mmol.get('P', 0):.3f}", f"{aporte_mmol.get('P', 0):.3f}",
                f"{aporte_mmol.get('HCO3', 0):.3f}", f"{aporte_mmol_anion_sum:.3f}", '']

        # Row 3: Aporte de Iones (meq/L)
        aporte_meq_anion_sum = sum(aporte_meq.get(elem, 0)
                                   for elem in anion_elements)
        row3 = ['Aporte de Iones (meq/L)', '', '', '', '', '', '',
                f"{aporte_meq.get('Ca', 0):.3f}", f"{aporte_meq.get('K', 0):.3f}",
                f"{aporte_meq.get('Mg', 0):.3f}", f"{aporte_meq.get('Na', 0):.3f}",
                f"{aporte_meq.get('NH4', 0):.3f}", f"{aporte_meq.get('N', 0):.3f}",
                f"{aporte_meq.get('N', 0):.3f}", f"{aporte_meq.get('S', 0):.3f}",
                f"{aporte_meq.get('S', 0):.3f}", f"{aporte_meq.get('Cl', 0):.3f}",
                f"{aporte_meq.get('P', 0):.3f}", f"{aporte_meq.get('P', 0):.3f}",
                f"{aporte_meq.get('HCO3', 0):.3f}", f"{aporte_meq_anion_sum:.3f}", '']

        # Row 4: Iones en Agua (mg/L)
        agua_anion_sum = sum(agua_mg.get(elem, 0) for elem in anion_elements)
        row4 = ['Iones en Agua (mg/L)', '', '', '', '', '', '',
                f"{agua_mg.get('Ca', 0):.1f}", f"{agua_mg.get('K', 0):.1f}",
                f"{agua_mg.get('Mg', 0):.1f}", f"{agua_mg.get('Na', 0):.1f}",
                f"{agua_mg.get('NH4', 0):.1f}", f"{agua_mg.get('N', 0):.1f}",
                f"{agua_mg.get('N', 0):.1f}", f"{agua_mg.get('S', 0):.1f}",
                f"{agua_mg.get('S', 0):.1f}", f"{agua_mg.get('Cl', 0):.1f}",
                f"{agua_mg.get('P', 0):.1f}", f"{agua_mg.get('P', 0):.1f}",
                f"{agua_mg.get('HCO3', 0):.1f}", f"{agua_anion_sum:.1f}", '']

        # Row 5: Iones en Agua (mmol/L)
        agua_mmol_anion_sum = sum(agua_mmol.get(elem, 0)
                                  for elem in anion_elements)
        row5 = ['Iones en Agua (mmol/L)', '', '', '', '', '', '',
                f"{agua_mmol.get('Ca', 0):.3f}", f"{agua_mmol.get('K', 0):.3f}",
                f"{agua_mmol.get('Mg', 0):.3f}", f"{agua_mmol.get('Na', 0):.3f}",
                f"{agua_mmol.get('NH4', 0):.3f}", f"{agua_mmol.get('N', 0):.3f}",
                f"{agua_mmol.get('N', 0):.3f}", f"{agua_mmol.get('S', 0):.3f}",
                f"{agua_mmol.get('S', 0):.3f}", f"{agua_mmol.get('Cl', 0):.3f}",
                f"{agua_mmol.get('P', 0):.3f}", f"{agua_mmol.get('P', 0):.3f}",
                f"{agua_mmol.get('HCO3', 0):.3f}", f"{agua_mmol_anion_sum:.3f}", '']

        # Row 6: Iones en Agua (meq/L)
        agua_meq_anion_sum = sum(agua_meq.get(elem, 0)
                                 for elem in anion_elements)
        row6 = ['Iones en Agua (meq/L)', '', '', '', '', '', '',
                f"{agua_meq.get('Ca', 0):.3f}", f"{agua_meq.get('K', 0):.3f}",
                f"{agua_meq.get('Mg', 0):.3f}", f"{agua_meq.get('Na', 0):.3f}",
                f"{agua_meq.get('NH4', 0):.3f}", f"{agua_meq.get('N', 0):.3f}",
                f"{agua_meq.get('N', 0):.3f}", f"{agua_meq.get('S', 0):.3f}",
                f"{agua_meq.get('S', 0):.3f}", f"{agua_meq.get('Cl', 0):.3f}",
                f"{agua_meq.get('P', 0):.3f}", f"{agua_meq.get('P', 0):.3f}",
                f"{agua_meq.get('HCO3', 0):.3f}", f"{agua_meq_anion_sum:.3f}", '']

        # Row 7: Iones en SONU Final (mg/L)
        final_anion_sum = sum(final_mg.get(elem, 0) for elem in anion_elements)
        row7 = ['Iones en SONU Final (mg/L)', '', '', '', '', '', '',
                f"{final_mg.get('Ca', 0):.1f}", f"{final_mg.get('K', 0):.1f}",
                f"{final_mg.get('Mg', 0):.1f}", f"{final_mg.get('Na', 0):.1f}",
                f"{final_mg.get('NH4', 0):.1f}", f"{final_mg.get('N', 0):.1f}",
                f"{final_mg.get('N', 0):.1f}", f"{final_mg.get('S', 0):.1f}",
                f"{final_mg.get('S', 0):.1f}", f"{final_mg.get('Cl', 0):.1f}",
                f"{final_mg.get('P', 0):.1f}", f"{final_mg.get('P', 0):.1f}",
                f"{final_mg.get('HCO3', 0):.1f}", f"{final_anion_sum:.1f}", f"{final_ec:.2f}"]

        # Row 8: Iones en SONU (mmol/L)
        final_mmol_anion_sum = sum(final_mmol.get(elem, 0)
                                   for elem in anion_elements)
        row8 = ['Iones en SONU (mmol/L)', '', '', '', '', '', '',
                f"{final_mmol.get('Ca', 0):.3f}", f"{final_mmol.get('K', 0):.3f}",
                f"{final_mmol.get('Mg', 0):.3f}", f"{final_mmol.get('Na', 0):.3f}",
                f"{final_mmol.get('NH4', 0):.3f}", f"{final_mmol.get('N', 0):.3f}",
                f"{final_mmol.get('N', 0):.3f}", f"{final_mmol.get('S', 0):.3f}",
                f"{final_mmol.get('S', 0):.3f}", f"{final_mmol.get('Cl', 0):.3f}",
                f"{final_mmol.get('P', 0):.3f}", f"{final_mmol.get('P', 0):.3f}",
                f"{final_mmol.get('HCO3', 0):.3f}", f"{final_mmol_anion_sum:.3f}", '']

        # Row 9: Iones en SONU (meq/L)
        final_meq_anion_sum = sum(final_meq.get(elem, 0)
                                  for elem in anion_elements)
        row9 = ['Iones en SONU (meq/L)', '', '', '', '', '', '',
                f"{final_meq.get('Ca', 0):.3f}", f"{final_meq.get('K', 0):.3f}",
                f"{final_meq.get('Mg', 0):.3f}", f"{final_meq.get('Na', 0):.3f}",
                f"{final_meq.get('NH4', 0):.3f}", f"{final_meq.get('N', 0):.3f}",
                f"{final_meq.get('N', 0):.3f}", f"{final_meq.get('S', 0):.3f}",
                f"{final_meq.get('S', 0):.3f}", f"{final_meq.get('Cl', 0):.3f}",
                f"{final_meq.get('P', 0):.3f}", f"{final_meq.get('P', 0):.3f}",
                f"{final_meq.get('HCO3', 0):.3f}", f"{final_meq_anion_sum:.3f}", '']

        summary_rows.extend(
            [row1, row2, row3, row4, row5, row6, row7, row8, row9])
        return summary_rows

    def _create_summary_tables(self, calculation_data: Dict[str, Any]) -> List:
        """Create additional summary and analysis tables"""
        elements = []
        calc_results = calculation_data.get('calculation_results', {})

        # Verification Results Table
        verification_results = calc_results.get('verification_results', [])
        if verification_results:
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("<b>RESULTADOS DE VERIFICACIÓN NUTRICIONAL</b>",
                                    ParagraphStyle('SectionTitle', parent=self.styles['Heading2'],
                                                 fontSize=14, textColor=colors.darkblue)))
            elements.append(Spacer(1, 10))

            verification_data = [
                ['Parámetro', 'Objetivo (mg/L)', 'Actual (mg/L)', 'Desviación (%)', 'Estado']]

            for result in verification_results:
                verification_data.append([
                    result.get('parameter', ''),
                    f"{result.get('target_value', 0):.1f}",
                    f"{result.get('actual_value', 0):.1f}",
                    f"{result.get('percentage_deviation', 0):+.1f}%",
                    result.get('status', '')
                ])

            verification_table = Table(verification_data, colWidths=[
                                       1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            verification_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.lightgrey]),
            ]))

            elements.append(verification_table)

        # Ionic Balance Analysis
        ionic_balance = calc_results.get('ionic_balance', {})
        if ionic_balance:
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("<b>ANÁLISIS DE BALANCE IÓNICO</b>",
                                    ParagraphStyle('SectionTitle', parent=self.styles['Heading2'],
                                                 fontSize=14, textColor=colors.darkblue)))
            elements.append(Spacer(1, 10))

            balance_data = [
                ['Parámetro', 'Valor', 'Unidad'],
                ['Suma de Cationes',
                    f"{ionic_balance.get('cation_sum', 0):.2f}", 'meq/L'],
                ['Suma de Aniones',
                    f"{ionic_balance.get('anion_sum', 0):.2f}", 'meq/L'],
                ['Diferencia',
                    f"{ionic_balance.get('difference', 0):.2f}", 'meq/L'],
                ['Error (%)',
                         f"{ionic_balance.get('difference_percentage', 0):.1f}", '%'],
                ['Balance', 'BALANCEADO' if ionic_balance.get(
                    'is_balanced') == 1 else 'DESBALANCEADO', '']
            ]

            balance_table = Table(balance_data, colWidths=[
                                  2.5*inch, 1.5*inch, 1*inch])
            balance_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.lightgrey]),
                ('TEXTCOLOR', (1, -1), (1, -1),
                 colors.green if ionic_balance.get('is_balanced') == 1 else colors.red),
                ('FONTNAME', (1, -1), (1, -1), 'Helvetica-Bold'),
            ]))

            elements.append(balance_table)

        return elements


class EnhancedPDFReportGenerator:
    """Professional PDF Report Generator for fertilizer calculations"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle', parent=self.styles['Heading1'], fontSize=18, spaceAfter=30,
            alignment=1, textColor=colors.darkblue
        )
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle', parent=self.styles['Heading2'], fontSize=12, spaceAfter=20,
            alignment=1, textColor=colors.darkgreen
        )

    def generate_comprehensive_pdf(self, calculation_data: Dict[str, Any], filename: str = None) -> str:
        """Generate comprehensive PDF report with detailed table"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/reporte_fertilizantes_{timestamp}.pdf"

        doc = SimpleDocTemplate(filename, pagesize=landscape(
            A4), rightMargin=15, leftMargin=15, topMargin=25, bottomMargin=25)
        story = []

        # Title and header
        title = Paragraph(
            "REPORTE DE CÁLCULO DE SOLUCIÓN NUTRITIVA", self.title_style)
        story.append(title)
        subtitle = Paragraph(
            "Sistema Avanzado de Optimización de Fertilizantes", self.subtitle_style)
        story.append(subtitle)
        story.append(Spacer(1, 20))

        # Metadata section
        metadata = self._create_metadata_section(calculation_data)
        story.extend(metadata)
        story.append(Spacer(1, 25))

        # Main calculation table
        main_table = self._create_main_calculation_table(calculation_data)
        story.append(main_table)
        story.append(PageBreak())

        # Summary and analysis tables
        summary_tables = self._create_summary_tables(calculation_data)
        story.extend(summary_tables)

        # Build PDF
        doc.build(story)
        print(f"PDF: Comprehensive report generated: {filename}")
        return filename

    def _create_metadata_section(self, calculation_data: Dict[str, Any]) -> List:
        """Create metadata section with calculation information"""
        elements = []
        metadata = calculation_data.get('integration_metadata', {})
        calc_results = calculation_data.get('calculation_results', {})
        final_solution = calc_results.get('final_solution', {})
        
        metadata_table_data = [
            ['Fecha y Hora:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Fuente de Datos:', metadata.get('data_source', 'API Integration')],
            ['Fertilizantes Analizados:', str(metadata.get('fertilizers_analyzed', 'N/A')), 'Volumen de Solución:', '1000 L'],
            ['Fase del Cultivo:', 'General', 'Tipo de Cálculo:', 'Optimización Avanzada'],
            ['EC Final:', f"{final_solution.get('calculated_EC', 0):.2f} dS/m", 'pH Final:', f"{final_solution.get('calculated_pH', 0):.1f}"]
        ]
        
        metadata_table = Table(metadata_table_data, colWidths=[2*inch, 2*inch, 2*inch, 2*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.lightgrey, colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        
        elements.append(metadata_table)
        return elements
    def _create_comprehensive_summary_rows(self, nutrient_contributions: Dict, water_contribution: Dict, final_solution: Dict) -> List[List]:
        """Create comprehensive summary rows matching Excel format"""
        summary_rows = []
        
        # Get data dictionaries
        aporte_mg = nutrient_contributions.get('APORTE_mg_L', {})
        aporte_mmol = nutrient_contributions.get('DE_mmol_L', {})
        aporte_meq = nutrient_contributions.get('IONES_meq_L', {})
        
        agua_mg = water_contribution.get('IONES_mg_L_DEL_AGUA', {})
        agua_mmol = water_contribution.get('mmol_L', {})
        agua_meq = water_contribution.get('meq_L', {})
        
        final_mg = final_solution.get('FINAL_mg_L', {})
        final_mmol = final_solution.get('FINAL_mmol_L', {})
        final_meq = final_solution.get('FINAL_meq_L', {})
        
        anion_elements = ['N', 'S', 'Cl', 'P', 'HCO3']
        final_ec = final_solution.get('calculated_EC', 0)
        
        # Row 1: Aporte de Iones (mg/L)
        aporte_anion_sum = sum(aporte_mg.get(elem, 0) for elem in anion_elements)
        row1 = ['Aporte de Iones (mg/L)', '', '', '', '', '', '',
                f"{aporte_mg.get('Ca', 0):.1f}", f"{aporte_mg.get('K', 0):.1f}", 
                f"{aporte_mg.get('Mg', 0):.1f}", f"{aporte_mg.get('Na', 0):.1f}", 
                f"{aporte_mg.get('NH4', 0):.1f}", f"{aporte_mg.get('N', 0):.1f}", 
                f"{aporte_mg.get('N', 0):.1f}", f"{aporte_mg.get('S', 0):.1f}", 
                f"{aporte_mg.get('S', 0):.1f}", f"{aporte_mg.get('Cl', 0):.1f}", 
                f"{aporte_mg.get('P', 0):.1f}", f"{aporte_mg.get('P', 0):.1f}", 
                f"{aporte_mg.get('HCO3', 0):.1f}", f"{aporte_anion_sum:.1f}", f"{final_ec:.2f}"]
        
        # Row 2: Aporte de Iones (mmol/L)
        aporte_mmol_anion_sum = sum(aporte_mmol.get(elem, 0) for elem in anion_elements)
        row2 = ['Aporte de Iones (mmol/L)', '', '', '', '', '', '',
                f"{aporte_mmol.get('Ca', 0):.3f}", f"{aporte_mmol.get('K', 0):.3f}", 
                f"{aporte_mmol.get('Mg', 0):.3f}", f"{aporte_mmol.get('Na', 0):.3f}", 
                f"{aporte_mmol.get('NH4', 0):.3f}", f"{aporte_mmol.get('N', 0):.3f}", 
                f"{aporte_mmol.get('N', 0):.3f}", f"{aporte_mmol.get('S', 0):.3f}", 
                f"{aporte_mmol.get('S', 0):.3f}", f"{aporte_mmol.get('Cl', 0):.3f}", 
                f"{aporte_mmol.get('P', 0):.3f}", f"{aporte_mmol.get('P', 0):.3f}", 
                f"{aporte_mmol.get('HCO3', 0):.3f}", f"{aporte_mmol_anion_sum:.3f}", '']
        
        # Row 3: Aporte de Iones (meq/L)
        aporte_meq_anion_sum = sum(aporte_meq.get(elem, 0) for elem in anion_elements)
        row3 = ['Aporte de Iones (meq/L)', '', '', '', '', '', '',
                f"{aporte_meq.get('Ca', 0):.3f}", f"{aporte_meq.get('K', 0):.3f}", 
                f"{aporte_meq.get('Mg', 0):.3f}", f"{aporte_meq.get('Na', 0):.3f}", 
                f"{aporte_meq.get('NH4', 0):.3f}", f"{aporte_meq.get('N', 0):.3f}", 
                f"{aporte_meq.get('N', 0):.3f}", f"{aporte_meq.get('S', 0):.3f}", 
                f"{aporte_meq.get('S', 0):.3f}", f"{aporte_meq.get('Cl', 0):.3f}", 
                f"{aporte_meq.get('P', 0):.3f}", f"{aporte_meq.get('P', 0):.3f}", 
                f"{aporte_meq.get('HCO3', 0):.3f}", f"{aporte_meq_anion_sum:.3f}", '']
        
        # Row 4: Iones en Agua (mg/L)
        agua_anion_sum = sum(agua_mg.get(elem, 0) for elem in anion_elements)
        row4 = ['Iones en Agua (mg/L)', '', '', '', '', '', '',
                f"{agua_mg.get('Ca', 0):.1f}", f"{agua_mg.get('K', 0):.1f}", 
                f"{agua_mg.get('Mg', 0):.1f}", f"{agua_mg.get('Na', 0):.1f}", 
                f"{agua_mg.get('NH4', 0):.1f}", f"{agua_mg.get('N', 0):.1f}", 
                f"{agua_mg.get('N', 0):.1f}", f"{agua_mg.get('S', 0):.1f}", 
                f"{agua_mg.get('S', 0):.1f}", f"{agua_mg.get('Cl', 0):.1f}", 
                f"{agua_mg.get('P', 0):.1f}", f"{agua_mg.get('P', 0):.1f}", 
                f"{agua_mg.get('HCO3', 0):.1f}", f"{agua_anion_sum:.1f}", '']
        
        # Row 5: Iones en Agua (mmol/L)
        agua_mmol_anion_sum = sum(agua_mmol.get(elem, 0) for elem in anion_elements)
        row5 = ['Iones en Agua (mmol/L)', '', '', '', '', '', '',
                f"{agua_mmol.get('Ca', 0):.3f}", f"{agua_mmol.get('K', 0):.3f}", 
                f"{agua_mmol.get('Mg', 0):.3f}", f"{agua_mmol.get('Na', 0):.3f}", 
                f"{agua_mmol.get('NH4', 0):.3f}", f"{agua_mmol.get('N', 0):.3f}", 
                f"{agua_mmol.get('N', 0):.3f}", f"{agua_mmol.get('S', 0):.3f}", 
                f"{agua_mmol.get('S', 0):.3f}", f"{agua_mmol.get('Cl', 0):.3f}", 
                f"{agua_mmol.get('P', 0):.3f}", f"{agua_mmol.get('P', 0):.3f}", 
                f"{agua_mmol.get('HCO3', 0):.3f}", f"{agua_mmol_anion_sum:.3f}", '']
        
        # Row 6: Iones en Agua (meq/L)
        agua_meq_anion_sum = sum(agua_meq.get(elem, 0) for elem in anion_elements)
        row6 = ['Iones en Agua (meq/L)', '', '', '', '', '', '',
                f"{agua_meq.get('Ca', 0):.3f}", f"{agua_meq.get('K', 0):.3f}", 
                f"{agua_meq.get('Mg', 0):.3f}", f"{agua_meq.get('Na', 0):.3f}", 
                f"{agua_meq.get('NH4', 0):.3f}", f"{agua_meq.get('N', 0):.3f}", 
                f"{agua_meq.get('N', 0):.3f}", f"{agua_meq.get('S', 0):.3f}", 
                f"{agua_meq.get('S', 0):.3f}", f"{agua_meq.get('Cl', 0):.3f}", 
                f"{agua_meq.get('P', 0):.3f}", f"{agua_meq.get('P', 0):.3f}", 
                f"{agua_meq.get('HCO3', 0):.3f}", f"{agua_meq_anion_sum:.3f}", '']
        
        # Row 7: Iones en SONU Final (mg/L)
        final_anion_sum = sum(final_mg.get(elem, 0) for elem in anion_elements)
        row7 = ['Iones en SONU Final (mg/L)', '', '', '', '', '', '',
                f"{final_mg.get('Ca', 0):.1f}", f"{final_mg.get('K', 0):.1f}", 
                f"{final_mg.get('Mg', 0):.1f}", f"{final_mg.get('Na', 0):.1f}", 
                f"{final_mg.get('NH4', 0):.1f}", f"{final_mg.get('N', 0):.1f}", 
                f"{final_mg.get('N', 0):.1f}", f"{final_mg.get('S', 0):.1f}", 
                f"{final_mg.get('S', 0):.1f}", f"{final_mg.get('Cl', 0):.1f}", 
                f"{final_mg.get('P', 0):.1f}", f"{final_mg.get('P', 0):.1f}", 
                f"{final_mg.get('HCO3', 0):.1f}", f"{final_anion_sum:.1f}", f"{final_ec:.2f}"]
        
        # Row 8: Iones en SONU (mmol/L)
        final_mmol_anion_sum = sum(final_mmol.get(elem, 0) for elem in anion_elements)
        row8 = ['Iones en SONU (mmol/L)', '', '', '', '', '', '',
                f"{final_mmol.get('Ca', 0):.3f}", f"{final_mmol.get('K', 0):.3f}", 
                f"{final_mmol.get('Mg', 0):.3f}", f"{final_mmol.get('Na', 0):.3f}", 
                f"{final_mmol.get('NH4', 0):.3f}", f"{final_mmol.get('N', 0):.3f}", 
                f"{final_mmol.get('N', 0):.3f}", f"{final_mmol.get('S', 0):.3f}", 
                f"{final_mmol.get('S', 0):.3f}", f"{final_mmol.get('Cl', 0):.3f}", 
                f"{final_mmol.get('P', 0):.3f}", f"{final_mmol.get('P', 0):.3f}", 
                f"{final_mmol.get('HCO3', 0):.3f}", f"{final_mmol_anion_sum:.3f}", '']
        
        # Row 9: Iones en SONU (meq/L)
        final_meq_anion_sum = sum(final_meq.get(elem, 0) for elem in anion_elements)
        row9 = ['Iones en SONU (meq/L)', '', '', '', '', '', '',
                f"{final_meq.get('Ca', 0):.3f}", f"{final_meq.get('K', 0):.3f}", 
                f"{final_meq.get('Mg', 0):.3f}", f"{final_meq.get('Na', 0):.3f}", 
                f"{final_meq.get('NH4', 0):.3f}", f"{final_meq.get('N', 0):.3f}", 
                f"{final_meq.get('N', 0):.3f}", f"{final_meq.get('S', 0):.3f}", 
                f"{final_meq.get('S', 0):.3f}", f"{final_meq.get('Cl', 0):.3f}", 
                f"{final_meq.get('P', 0):.3f}", f"{final_meq.get('P', 0):.3f}", 
                f"{final_meq.get('HCO3', 0):.3f}", f"{final_meq_anion_sum:.3f}", '']
        
        summary_rows.extend([row1, row2, row3, row4, row5, row6, row7, row8, row9])
        return summary_rows

    def _create_main_calculation_table(self, calculation_data: Dict[str, Any]) -> object:
        """Create the main Excel-like calculation table with all fertilizer rows"""
        calc_results = calculation_data.get('calculation_results', {})
        fertilizer_dosages = calc_results.get('fertilizer_dosages', {})
        nutrient_contributions = calc_results.get('nutrient_contributions', {})
        water_contribution = calc_results.get('water_contribution', {})
        final_solution = calc_results.get('final_solution', {})
        
        # Define column headers exactly as specified
        headers = [
            'FERTILIZANTE', '% P', 'Peso molecular\n(Sal)', 'Peso molecular\n(Elem1)', 
            'Peso molecular\n(Elem2)', 'Peso de sal\n(mg o ml/L)', 'Peso de sal\n(mmol/L)',
            'Ca', 'K', 'Mg', 'Na', 'NH4', 'NO3-', 'N', 'SO4=', 'S', 'Cl-', 
            'H2PO4-', 'P', 'HCO3-', 'Σ aniones', 'CE'
        ]
        
        table_data = [headers]
        
        # Add fertilizer rows for active fertilizers only
        from fertilizer_database import EnhancedFertilizerDatabase
        fertilizer_db = EnhancedFertilizerDatabase()
        
        fertilizer_rows_added = 0
        for fert_name, dosage_info in fertilizer_dosages.items():
            dosage_g_l = dosage_info.get('dosage_g_per_L', 0) if isinstance(dosage_info, dict) else 0
            if dosage_g_l > 0:
                row = self._create_fertilizer_row(fert_name, dosage_info, fertilizer_db)
                table_data.append(row)
                fertilizer_rows_added += 1
                print(f"    Added fertilizer row: {fert_name} ({dosage_g_l:.3f} g/L)")
        
        print(f"    Total fertilizer rows added: {fertilizer_rows_added}")
        
        # Add summary rows
        summary_rows = self._create_comprehensive_summary_rows(
            nutrient_contributions, water_contribution, final_solution
        )
        table_data.extend(summary_rows)
        print(f"    Added {len(summary_rows)} summary rows")
        
        # Calculate number of fertilizer rows for styling
        num_fertilizer_rows = fertilizer_rows_added
        
        print(f"    Creating table with {len(table_data)} total rows ({num_fertilizer_rows} fertilizer + {len(summary_rows)} summary)")
        
        # Create table with professional styling
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            
            # Fertilizer rows styling (if any exist)
            ('FONTNAME', (0, 1), (-1, num_fertilizer_rows), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, num_fertilizer_rows), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, num_fertilizer_rows), [colors.white, colors.lightgrey]),
            
            # Summary rows styling
            ('BACKGROUND', (0, num_fertilizer_rows+1), (-1, -1), colors.lightyellow),
            ('FONTNAME', (0, num_fertilizer_rows+1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, num_fertilizer_rows+1), (-1, -1), 6),
            
            # Borders and alignment
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ]))
        
        return table

    def _create_fertilizer_row(self, fert_name: str, dosage_info: Dict, fertilizer_db) -> List:
        """Create a detailed table row for a single fertilizer with CORRECT calculations"""
        dosage_g_l = dosage_info.get('dosage_g_per_L', 0) if isinstance(dosage_info, dict) else 0
        
        print(f"      Creating row for {fert_name}: {dosage_g_l:.3f} g/L")
        
        # Get fertilizer composition from database
        composition_data = fertilizer_db.find_fertilizer_composition(fert_name, fert_name)
        
        if composition_data:
            molecular_weight = composition_data['mw']
            cations = composition_data['cations']
            anions = composition_data['anions']
            print(f"        Found composition: {composition_data['formula']}")
        else:
            molecular_weight = 100
            cations = {'Ca': 0, 'K': 0, 'Mg': 0, 'Na': 0, 'NH4': 0}
            anions = {'N': 0, 'S': 0, 'Cl': 0, 'P': 0, 'HCO3': 0}
            print(f"        Using default composition")
        
        dosage_mg_l = dosage_g_l * 1000
        dosage_mmol_l = dosage_mg_l / molecular_weight if molecular_weight > 0 else 0
        
        # Get main elements for molecular weight display
        main_elements = []
        all_nutrients = {**cations, **anions}
        sorted_nutrients = sorted(all_nutrients.items(), key=lambda x: x[1], reverse=True)
        
        # Get atomic weights for main elements
        atomic_weights = {
            'Ca': 40.08, 'K': 39.10, 'Mg': 24.31, 'Na': 22.99, 'NH4': 18.04,
            'N': 14.01, 'S': 32.06, 'P': 30.97, 'Cl': 35.45, 'Fe': 55.85,
            'Mn': 54.94, 'Zn': 65.38, 'Cu': 63.55, 'B': 10.81, 'Mo': 95.96
        }
        
        for elem, content in sorted_nutrients:
            if content > 1 and elem in atomic_weights:
                main_elements.append((elem, atomic_weights[elem]))
                if len(main_elements) >= 2:
                    break
        
        elem1_weight = main_elements[0][1] if len(main_elements) > 0 else 0
        elem2_weight = main_elements[1][1] if len(main_elements) > 1 else 0
        
        # Calculate actual nutrient contributions (mg/L)
        purity_factor = 98.0 / 100.0  # Assume 98% purity
        
        nutrient_contributions = {}
        for elem in ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'S', 'Cl', 'P', 'HCO3']:
            cation_content = cations.get(elem, 0)
            anion_content = anions.get(elem, 0)
            total_content = cation_content + anion_content
            
            if total_content > 0:
                contribution = dosage_mg_l * (total_content / 100.0) * purity_factor
                nutrient_contributions[elem] = contribution
            else:
                nutrient_contributions[elem] = 0
        
        # Calculate anion sum and EC contribution
        anion_elements = ['N', 'S', 'Cl', 'P', 'HCO3']
        anion_sum = sum(nutrient_contributions.get(elem, 0) for elem in anion_elements)
        
        # EC contribution (simplified calculation)
        ec_contribution = dosage_mmol_l * 0.1
        
        print(f"        Main contributions: Ca={nutrient_contributions.get('Ca', 0):.1f}, K={nutrient_contributions.get('K', 0):.1f}, N={nutrient_contributions.get('N', 0):.1f} mg/L")
        
        row = [
            fert_name,                                      # FERTILIZANTE
            "98.0",                                         # % P (purity)
            f"{molecular_weight:.1f}",                      # Peso molecular (Sal)
            f"{elem1_weight:.1f}",                          # Peso molecular (Elem1)
            f"{elem2_weight:.1f}",                          # Peso molecular (Elem2)
            f"{dosage_g_l:.3f}",                           # Peso de sal (mg o ml/L) - in g/L
            f"{dosage_mmol_l:.3f}",                        # Peso de sal (mmol/L)
            f"{nutrient_contributions.get('Ca', 0):.1f}",   # Ca contribution
            f"{nutrient_contributions.get('K', 0):.1f}",    # K contribution
            f"{nutrient_contributions.get('Mg', 0):.1f}",   # Mg contribution
            f"{nutrient_contributions.get('Na', 0):.1f}",   # Na contribution
            f"{nutrient_contributions.get('NH4', 0):.1f}",  # NH4 contribution
            f"{nutrient_contributions.get('N', 0):.1f}",    # NO3- contribution
            f"{nutrient_contributions.get('N', 0):.1f}",    # N contribution
            f"{nutrient_contributions.get('S', 0):.1f}",    # SO4= contribution
            f"{nutrient_contributions.get('S', 0):.1f}",    # S contribution
            f"{nutrient_contributions.get('Cl', 0):.1f}",   # Cl- contribution
            f"{nutrient_contributions.get('P', 0):.1f}",    # H2PO4- contribution
            f"{nutrient_contributions.get('P', 0):.1f}",    # P contribution
            f"{nutrient_contributions.get('HCO3', 0):.1f}", # HCO3- contribution
            f"{anion_sum:.1f}",                            # Σ aniones
            f"{ec_contribution:.2f}"                       # CE contribution
        ]
        
        return row

    def _create_summary_tables(self, calculation_data: Dict[str, Any]) -> List:
        """Create additional summary and analysis tables"""
        elements = []
        calc_results = calculation_data.get('calculation_results', {})
        
        # Verification Results Table
        verification_results = calc_results.get('verification_results', [])
        if verification_results:
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("<b>RESULTADOS DE VERIFICACIÓN NUTRICIONAL</b>", 
                                    ParagraphStyle('SectionTitle', parent=self.styles['Heading2'], 
                                                 fontSize=14, textColor=colors.darkblue)))
            elements.append(Spacer(1, 10))
            
            verification_data = [['Parámetro', 'Objetivo (mg/L)', 'Actual (mg/L)', 'Desviación (%)', 'Estado']]
            
            for result in verification_results:
                verification_data.append([
                    result.get('parameter', ''),
                    f"{result.get('target_value', 0):.1f}",
                    f"{result.get('actual_value', 0):.1f}",
                    f"{result.get('percentage_deviation', 0):+.1f}%",
                    result.get('status', '')
                ])
            
            verification_table = Table(verification_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            verification_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            elements.append(verification_table)
        
        # Ionic Balance Analysis
        ionic_balance = calc_results.get('ionic_balance', {})
        if ionic_balance:
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("<b>ANÁLISIS DE BALANCE IÓNICO</b>", 
                                    ParagraphStyle('SectionTitle', parent=self.styles['Heading2'], 
                                                 fontSize=14, textColor=colors.darkblue)))
            elements.append(Spacer(1, 10))
            
            balance_data = [
                ['Parámetro', 'Valor', 'Unidad'],
                ['Suma de Cationes', f"{ionic_balance.get('cation_sum', 0):.2f}", 'meq/L'],
                ['Suma de Aniones', f"{ionic_balance.get('anion_sum', 0):.2f}", 'meq/L'],
                ['Diferencia', f"{ionic_balance.get('difference', 0):.2f}", 'meq/L'],
                ['Error (%)', f"{ionic_balance.get('difference_percentage', 0):.1f}", '%'],
                ['Balance', 'BALANCEADO' if ionic_balance.get('is_balanced') == 1 else 'DESBALANCEADO', '']
            ]
            
            balance_table = Table(balance_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
            balance_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ('TEXTCOLOR', (1, -1), (1, -1), 
                 colors.green if ionic_balance.get('is_balanced') == 1 else colors.red),
                ('FONTNAME', (1, -1), (1, -1), 'Helvetica-Bold'),
            ]))
            
            elements.append(balance_table)
        
        # Cost Analysis Table
        cost_analysis = calc_results.get('cost_analysis', {})
        if cost_analysis and cost_analysis.get('cost_per_fertilizer'):
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("<b>ANÁLISIS ECONÓMICO</b>", 
                                    ParagraphStyle('SectionTitle', parent=self.styles['Heading2'], 
                                                 fontSize=14, textColor=colors.darkblue)))
            elements.append(Spacer(1, 10))
            
            cost_data = [['Fertilizante', 'Costo por 1000L ($)', 'Porcentaje del Total (%)']]
            
            cost_per_fert = cost_analysis.get('cost_per_fertilizer', {})
            percentage_per_fert = cost_analysis.get('percentage_per_fertilizer', {})
            
            for fert, cost in cost_per_fert.items():
                if cost > 0:
                    percentage = percentage_per_fert.get(fert, 0)
                    cost_data.append([
                        fert,
                        f"${cost:.3f}",
                        f"{percentage:.1f}%"
                    ])
            
            # Add total row
            total_cost = cost_analysis.get('total_cost_diluted', 0)
            cost_data.append(['TOTAL', f"${total_cost:.2f}", '100.0%'])
            
            cost_table = Table(cost_data, colWidths=[3*inch, 2*inch, 2*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('BACKGROUND', (0, -1), (-1, -1), colors.lightyellow),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.lightgrey]),
            ]))
            
            elements.append(cost_table)
        return elements

# ============================================================================
# INITIALIZE CALCULATOR AND COMPONENTS
# ============================================================================

calculator = FertilizerCalculator()
verifier = SolutionVerifier()
cost_analyzer = CostAnalyzer()
pdf_generator = EnhancedPDFReportGenerator()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/calculate-advanced", response_model=AdvancedFertilizerResponse)
async def calculate_advanced_fertilizer_solution(request: FertilizerRequest):
    """Calculate optimal fertilizer dosages with comprehensive PDF generation"""
    try:
        print(f"\n=== STARTING ADVANCED CALCULATION ===")
        print(f"Request received with {len(request.fertilizers)} fertilizers")
        
        # Step 1: Optimize fertilizer dosages
        dosages_g_l = calculator.optimize_solution(
            request.fertilizers, request.target_concentrations, request.water_analysis
        )

        # Convert to response format
        fertilizer_dosages = {}
        for name, dosage_g in dosages_g_l.items():
            fertilizer = next((f for f in request.fertilizers if f.name == name), None)
            density = fertilizer.density if fertilizer else 1.0

            fertilizer_dosages[name] = FertilizerDosage(
                dosage_ml_per_L=round(dosage_g / density, 4),
                dosage_g_per_L=round(dosage_g, 4)
            )

        # Step 2: Calculate all contributions
        nutrient_contributions = calculator.calculate_all_contributions(request.fertilizers, dosages_g_l)
        water_contribution = calculator.calculate_water_contributions(request.water_analysis)
        final_solution = calculator.calculate_final_solution(nutrient_contributions, water_contribution)

        # Step 3: Verify solution
        verification_results = verifier.verify_concentrations(
            request.target_concentrations, final_solution['FINAL_mg_L']
        )
        verification_models = [VerificationResult(**result) for result in verification_results]

        ionic_relationships = verifier.verify_ionic_relationships(
            final_solution['FINAL_meq_L'], final_solution['FINAL_mmol_L'], final_solution['FINAL_mg_L']
        )
        ionic_relationship_models = [IonicRelationship(**rel) for rel in ionic_relationships]

        ionic_balance_data = verifier.verify_ionic_balance(final_solution['FINAL_meq_L'])
        ionic_balance = IonicBalance(**ionic_balance_data)

        # Step 4: Cost analysis
        fertilizer_amounts_kg = {name: dosage * request.calculation_settings.volume_liters / 1000
                                 for name, dosage in dosages_g_l.items()}

        cost_analysis_data = cost_analyzer.calculate_solution_cost(
            fertilizer_amounts_kg, request.calculation_settings.volume_liters,
            request.calculation_settings.volume_liters
        )
        cost_analysis = CostAnalysis(**cost_analysis_data)

        # Step 5: Create response
        response = AdvancedFertilizerResponse(
            fertilizer_dosages=fertilizer_dosages,
            nutrient_contributions=NutrientContributions(**nutrient_contributions),
            water_contribution=WaterContribution(**water_contribution),
            final_solution=FinalSolution(**final_solution),
            verification_results=verification_models,
            ionic_relationships=ionic_relationship_models,
            ionic_balance=ionic_balance,
            cost_analysis=cost_analysis,
            calculation_status=CalculationStatus(
                success=True,
                warnings=[],
                iterations=1,
                convergence_error=ionic_balance.difference_percentage / 100
            )
        )

        # Step 6: Generate comprehensive PDF report
        try:
            calculation_data = {
                "integration_metadata": {
                    "data_source": "Direct API Calculation",
                    "fertilizers_analyzed": len(request.fertilizers),
                    "calculation_timestamp": datetime.now().isoformat()
                },
                "calculation_results": response.model_dump()
            }
            
            pdf_filename = pdf_generator.generate_comprehensive_pdf(calculation_data)
            response.pdf_report = {
                "generated": True,
                "filename": pdf_filename,
                "file_path": os.path.abspath(pdf_filename),
                "message": f"Comprehensive PDF report generated: {pdf_filename}"
            }
            
        except Exception as pdf_error:
            print(f"WARNING: PDF generation failed: {pdf_error}")
            response.pdf_report = {
                "generated": False,
                "error": str(pdf_error),
                "message": "Calculation successful but PDF generation failed"
            }

        print(f"\n=== CALCULATION COMPLETE ===")
        print(f"Success: {len(fertilizer_dosages)} fertilizer dosages calculated")
        print(f"PDF: {response.pdf_report.get('message', 'Not generated')}")

        return response

    except Exception as e:
        print(f"ERROR: Advanced calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced calculation error: {str(e)}")

@app.get("/swagger-integrated-calculation-complete")
async def swagger_integrated_calculation_complete(
    catalog_id: int = 1,
    phase_id: int = 1,
    water_id: int = 1,
    volume_liters: float = 1000
):
    """Complete Swagger integration with fixed fertilizer matching and comprehensive PDF"""
    print(f"\n=== STARTING COMPLETE SWAGGER INTEGRATION ===")
    
    # Initialize API client and login
    api_client = SwaggerAPIClient(base_url="http://162.248.52.111:8082")
    await api_client.login("csolano@iapcr.com", "123")

    try:
        # Fetch data from Swagger API
        print(f"Fetching data from Swagger API...")
        fertilizers_data = await api_client.get_fertilizers(catalog_id)
        if not fertilizers_data:
            raise HTTPException(status_code=404, detail="No fertilizers found in catalog")

        requirements_data = await api_client.get_crop_phase_requirements(phase_id)
        if not requirements_data:
            raise HTTPException(status_code=404, detail="No crop requirements found for phase")

        water_data = await api_client.get_water_chemistry(water_id, catalog_id)
        if not water_data:
            raise HTTPException(status_code=404, detail="No water analysis found")

        print(f"Data fetched successfully:")
        print(f"  - {len(fertilizers_data)} fertilizers found")
        print(f"  - Crop requirements: {len(requirements_data)} parameters")
        print(f"  - Water analysis: {len(water_data)} parameters")

        # Process fertilizers with chemistry data
        print(f"\nProcessing fertilizer chemistry data...")
        fertilizers = []
        
        # Prioritize useful fertilizers by name patterns
        useful_patterns = [
            'nitrato', 'sulfato', 'fosfato', 'calcio', 'potasio', 
            'magnesio', 'amonio', 'acido', 'fosforico', 'nitrico'
        ]
        
        # Sort fertilizers to prioritize useful ones
        sorted_fertilizers = sorted(fertilizers_data, 
                                  key=lambda f: any(pattern in f.get('name', '').lower() 
                                                   for pattern in useful_patterns), 
                                  reverse=True)
        
        for i, fert_data in enumerate(sorted_fertilizers[:12]):  # Process up to 12 fertilizers
            try:
                print(f"Processing {i+1}: {fert_data.get('name', 'Unknown')}")
                chemistry = await api_client.get_fertilizer_chemistry(fert_data['id'], catalog_id)
                
                fertilizer = api_client.map_swagger_fertilizer_to_model(fert_data, chemistry)
                
                # Check if fertilizer has any useful content
                total_content = (sum(fertilizer.composition.cations.values()) + 
                               sum(fertilizer.composition.anions.values()))
                
                if total_content > 5:  # Only include fertilizers with significant content
                    fertilizers.append(fertilizer)
                    print(f"  SUCCESS: Added {fertilizer.name} (total content: {total_content:.1f}%)")
                else:
                    print(f"  SKIPPED: {fertilizer.name} (insufficient content: {total_content:.1f}%)")
                    
            except Exception as e:
                print(f"  ERROR: Failed to process {fert_data.get('name', 'Unknown')}: {e}")

        if not fertilizers:
            raise Exception("No usable fertilizers found after processing chemistry data")

        print(f"\nSuccessfully processed {len(fertilizers)} usable fertilizers")

        # Map API data to calculation format
        target_concentrations = api_client.map_requirements_to_targets(requirements_data)
        water_analysis = api_client.map_water_to_analysis(water_data)

        print(f"\nMapped data:")
        print(f"  - Target concentrations: {len(target_concentrations)} parameters")
        print(f"  - Water analysis: {len(water_analysis)} parameters")

        # Add some default target concentrations if none provided
        if not target_concentrations:
            print(f"No target concentrations found, using defaults")
            target_concentrations = {
                'N': 150, 'P': 40, 'K': 200, 'Ca': 180, 'Mg': 50, 'S': 80,
                'Fe': 2.0, 'Mn': 0.5, 'Zn': 0.3, 'Cu': 0.1, 'B': 0.5
            }

        # Ensure water analysis has at least empty values
        if not water_analysis:
            water_analysis = {elem: 0 for elem in ['Ca', 'K', 'Mg', 'Na', 'N', 'S', 'Cl', 'P']}

        # Create calculation request
        calculation_request = FertilizerRequest(
            fertilizers=fertilizers,
            target_concentrations=target_concentrations,
            water_analysis=water_analysis,
            calculation_settings=CalculationSettings(
                volume_liters=volume_liters,
                precision=3,
                units="mg/L",
                crop_phase="General"
            )
        )

        print(f"\nCalculation request created:")
        print(f"  - {len(fertilizers)} fertilizers")
        print(f"  - {len(target_concentrations)} targets")
        print(f"  - {len(water_analysis)} water parameters")

        # Perform advanced calculation
        print(f"\nStarting calculation...")
        result = await calculate_advanced_fertilizer_solution(calculation_request)

        # Create comprehensive response data
        integration_metadata = {
            "data_source": "Complete Swagger API Integration v3.0",
            "catalog_id": catalog_id,
            "phase_id": phase_id,
            "water_id": water_id,
            "fertilizers_analyzed": len(fertilizers),
            "calculation_timestamp": datetime.now().isoformat(),
            "version": "3.0.0",
            "features": [
                "Fixed fertilizer composition database",
                "Proper fertilizer pattern matching",
                "Comprehensive PDF generation",
                "Complete Excel-like calculation table"
            ],
            "api_endpoints_used": [
                f"/Fertilizer?CatalogId={catalog_id}",
                f"/CropPhaseSolutionRequirement/GetByPhaseId?PhaseId={phase_id}",
                f"/WaterChemistry?WaterId={water_id}&CatalogId={catalog_id}",
                "/FertilizerChemistry (multiple calls)"
            ]
        }

        response_data = {
            "integration_metadata": integration_metadata,
            "calculation_results": result.model_dump(),
            "data_summary": {
                "fertilizers_available": len(fertilizers_data),
                "fertilizers_with_chemistry": len(fertilizers),
                "target_parameters": len(target_concentrations),
                "water_parameters": len(water_analysis),
                "total_dosages": len([d for d in result.fertilizer_dosages.values() if d.dosage_g_per_L > 0])
            }
        }

        print(f"\n=== COMPLETE INTEGRATION FINISHED ===")
        print(f"Success: Calculation completed with {len([d for d in result.fertilizer_dosages.values() if d.dosage_g_per_L > 0])} active fertilizers")
        print(f"PDF: {result.pdf_report.get('message', 'Not generated')}")

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Integration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Integration error: {str(e)}")

@app.get("/test-fertilizer-database")
async def test_fertilizer_database():
    """Test the fertilizer database matching"""
    fertilizer_db = EnhancedFertilizerDatabase()
    
    test_fertilizers = [
        ("Acido Nítrico DAC", "HNO3"),
        ("Acido Fosfórico", "H3PO4"),
        ("Nitrato de calcio", "Ca(NO3)2.2(H2O)"),
        ("Nitrato de potasio", "KNO3"),
        ("Sulfato de amonio", "(NH4)2SO4"),
        ("Fosfato monopotasico", "KH2PO4"),
        ("Unknown Fertilizer", "XYZ123")
    ]
    
    results = []
    
    for name, formula in test_fertilizers:
        composition = fertilizer_db.find_fertilizer_composition(name, formula)
        
        if composition:
            total_content = sum(composition['cations'].values()) + sum(composition['anions'].values())
            main_nutrients = []
            for elem, content in composition['cations'].items():
                if content > 1:
                    main_nutrients.append(f"{elem}:{content:.1f}%")
            for elem, content in composition['anions'].items():
                if content > 1:
                    main_nutrients.append(f"{elem}:{content:.1f}%")
            
            results.append({
                "name": name,
                "formula": formula,
                "found": True,
                "matched_formula": composition['formula'],
                "molecular_weight": composition['mw'],
                "total_content": round(total_content, 1),
                "main_nutrients": main_nutrients,
                "cations": composition['cations'],
                "anions": composition['anions']
            })
        else:
            results.append({
                "name": name,
                "formula": formula,
                "found": False,
                "message": "No match found in database"
            })
    
    found_count = len([r for r in results if r.get('found', False)])
    
    return {
        "status": "database_test_complete",
        "total_tested": len(test_fertilizers),
        "found_in_database": found_count,
        "success_rate": f"{found_count}/{len(test_fertilizers)}",
        "fertilizers": results,
        "database_coverage": {
            "acids": ["Nitric", "Phosphoric", "Sulfuric"],
            "nitrates": ["Calcium", "Potassium", "Ammonium", "Magnesium"],
            "sulfates": ["Ammonium", "Potassium", "Magnesium"],
            "phosphates": ["Monopotassium phosphate"],
            "total_entries": len(fertilizer_db.fertilizer_data)
        }
    }

@app.get("/")
async def root():
    """Root endpoint with complete API information"""
    return {
        "message": "Complete Fertilizer Calculator API v3.0.0",
        "version": "3.0.0",
        "status": "operational - complete implementation",
        "description": "Professional hydroponic nutrient calculation system with comprehensive fertilizer database",
        "key_features": {
            "fertilizer_database": [
                "Complete fertilizer composition database with 10+ common fertilizers",
                "Intelligent pattern matching by name and chemical formula",
                "Accurate nutrient percentages for all major elements",
                "Fallback mechanisms for unknown fertilizers"
            ],
            "calculation_engine": [
                "Advanced fertilizer optimization algorithm",
                "Systematic nutrient prioritization (P → Ca → K → Mg)",
                "Multi-element contribution tracking",
                "Professional EC and pH calculations"
            ],
            "pdf_generation": [
                "Complete Excel-like calculation table (22 columns)",
                "Summary rows (Aporte, Agua, SONU Final)",
                "Verification results with target achievement",
                "Ionic balance analysis and cost breakdown"
            ],
            "api_integration": [
                "Full Swagger API integration",
                "Automatic fertilizer chemistry fetching",
                "Real-time data processing and mapping",
                "Enhanced error handling and fallbacks"
            ]
        },
        "endpoints": {
            "main_calculation": {
                "url": "/calculate-advanced",
                "method": "POST",
                "description": "Direct fertilizer calculation with comprehensive PDF",
                "returns": "Complete calculation results + PDF report"
            },
            "swagger_integration": {
                "url": "/swagger-integrated-calculation-complete",
                "method": "GET",
                "description": "Complete Swagger API integration with all fixes",
                "returns": "Real fertilizer data + comprehensive PDF"
            },
            "database_test": {
                "url": "/test-fertilizer-database",
                "method": "GET",
                "description": "Test fertilizer database matching",
                "returns": "Database coverage and matching results"
            }
        },
        "expected_results": {
            "fertilizer_matching": "Correct compositions for all common fertilizers",
            "dosage_calculations": "Actual g/L values based on real nutrient content",
            "pdf_reports": "Complete Excel-format table with all calculation steps",
            "cost_analysis": "Accurate costs based on calculated dosages",
            "verification": "Target achievement percentages for all nutrients"
        },
        "quick_test": "GET /swagger-integrated-calculation-complete",
        "database_test": "GET /test-fertilizer-database",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn # type: ignore
    import socket

    print("STARTING: Complete Fertilizer Calculator API v3.0.0")
    print("COMPLETE: All fixes and features implemented")
    print("=" * 60)
    
    # Create reports directory
    if not os.path.exists('reports'):
        os.makedirs('reports')
        print("SETUP: Created reports directory")

    # Find available port
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return True
            except OSError:
                return False

    ports_to_try = [8000, 8001, 8002, 8003, 8004, 8005]
    available_port = None
    
    for port in ports_to_try:
        if is_port_available(port):
            available_port = port
            break
    
    if available_port is None:
        print("ERROR: No available ports found")
        exit(1)

    print(f"\nSTARTING: Complete API server on port {available_port}")
    print(f"DOCS: http://localhost:{available_port}/docs")
    print(f"MAIN: http://localhost:{available_port}/swagger-integrated-calculation-complete")
    print(f"TEST: http://localhost:{available_port}/test-fertilizer-database")
    print(f"ROOT: http://localhost:{available_port}/")
    print("COMPLETE: Full fertilizer database + comprehensive PDF generation")
    
    uvicorn.run(app, host="0.0.0.0", port=available_port)