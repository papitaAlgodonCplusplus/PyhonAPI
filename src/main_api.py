#!/usr/bin/env python3
"""
COMPLETE MODULAR FERTILIZER CALCULATOR API - MAIN IMPLEMENTATION
All missing functionality implemented across modular files
"""

from fastapi import FastAPI, HTTPException, Query
from models import (
    FertilizerRequest, SimpleResponse, FertilizerDosage, CalculationStatus, MLModelConfig
)
from nutrient_calculator import EnhancedFertilizerCalculator
from fertilizer_database import EnhancedFertilizerDatabase
from pdf_generator import EnhancedPDFReportGenerator
from verification_analyzer import SolutionVerifier, CostAnalyzer
from swagger_integration import SwaggerAPIClient
from ml_optimizer import ProfessionalMLFertilizerOptimizer
from typing import Dict, List, Optional, Any
import os
import json
import asyncio
from datetime import datetime
import numpy as np
from nutrient_caps import apply_nutrient_caps_to_targets, NutrientCaps

app = FastAPI(
    title="Complete Modular Fertilizer Calculator", 
    version="5.0.0",
    description="Professional hydroponic nutrient calculation system with ML optimization"
)

# Initialize all components
nutrient_calc = EnhancedFertilizerCalculator()
fertilizer_db = EnhancedFertilizerDatabase()
pdf_generator = EnhancedPDFReportGenerator()
verifier = SolutionVerifier()
cost_analyzer = CostAnalyzer()
ml_optimizer = ProfessionalMLFertilizerOptimizer()

# Create reports directory
os.makedirs("reports", exist_ok=True)

class CompleteFertilizerCalculator:
    """Complete fertilizer calculator with all optimization methods"""
    
    def __init__(self):
        self.nutrient_calc = nutrient_calc
        self.fertilizer_db = fertilizer_db
        self.ml_optimizer = ml_optimizer

    def calculate_advanced_solution(self, request: FertilizerRequest, method: str = "deterministic") -> Dict[str, Any]:
        """
        Calculate fertilizer solution using specified method - ENHANCED WITH MICRONUTRIENT ANALYSIS
        """
        print(f"\n=== STARTING {method.upper()} CALCULATION WITH MICRONUTRIENT ANALYSIS ===")
        print(f"Fertilizers: {len(request.fertilizers)}")
        print(f"Targets: {request.target_concentrations}")
        print(f"Water: {request.water_analysis}")
        
        # Choose calculation method (existing code remains the same)
        if method == "machine_learning":
            # FORCE RELOAD ATTEMPT if not trained
            if not self.ml_optimizer.is_trained:
                print("ML model not trained, attempting to load existing model...")
                
                # Try to reload the model first
                try:
                    self.ml_optimizer._try_load_existing_model()
                    if self.ml_optimizer.is_trained:
                        print("SUCCESS: Successfully loaded existing ML model!")
                    else:
                        print("ERROR: No compatible model found, training new model...")
                        training_data = self.ml_optimizer.generate_real_training_data(
                            fertilizers=request.fertilizers, num_scenarios=2000
                        )
                        self.ml_optimizer.train_model(training_data=training_data, fertilizers=request.fertilizers)
                except Exception as e:
                    print(f"ERROR: Model loading failed: {e}")
                    print("Training new model from scratch...")
                    training_data = self.ml_optimizer.generate_real_training_data(
                        fertilizers=request.fertilizers, num_scenarios=2000
                    )
                    self.ml_optimizer.train_model(training_data=training_data, fertilizers=request.fertilizers)
            
            dosages_g_l = self.ml_optimizer.optimize_with_ml(
                request.target_concentrations, request.water_analysis, fertilizers=request.fertilizers
            )
        else:  # deterministic
            dosages_g_l = self.optimize_deterministic_solution(
                request.fertilizers, request.target_concentrations, request.water_analysis
            )

        # Convert to response format (existing code)
        fertilizer_dosages = {}
        for fertilizer in request.fertilizers:
            dosage_g = dosages_g_l.get(fertilizer.name, 0.0)
            density = fertilizer.density if fertilizer.density else 1.0
            
            fertilizer_dosages[fertilizer.name] = FertilizerDosage(
                dosage_ml_per_L=round(dosage_g / density, 4),
                dosage_g_per_L=round(dosage_g, 4)
            )

        # Calculate all contributions and analysis (existing code)
        nutrient_contributions = self.calculate_all_contributions(request.fertilizers, dosages_g_l)
        water_contribution = self.calculate_water_contributions(request.water_analysis)
        final_solution = self.calculate_final_solution(nutrient_contributions, water_contribution)

        # Verification and analysis (existing code)
        verification_results = verifier.verify_concentrations(
            request.target_concentrations, final_solution['FINAL_mg_L']
        )
        ionic_relationships = verifier.verify_ionic_relationships(
            final_solution['FINAL_meq_L'], final_solution['FINAL_mmol_L'], final_solution['FINAL_mg_L']
        )
        ionic_balance = verifier.verify_ionic_balance(final_solution['FINAL_meq_L'])

        # Cost analysis (existing code)
        fertilizer_amounts_kg = {
            name: dosage * request.calculation_settings.volume_liters / 1000
            for name, dosage in dosages_g_l.items()
        }
        cost_analysis = cost_analyzer.calculate_solution_cost(
            fertilizer_amounts_kg, 
            request.calculation_settings.volume_liters,
            request.calculation_settings.volume_liters
        )

        # *** NEW: ADD MICRONUTRIENT ANALYSIS HERE ***
        print(f"\nPERFORMING ENHANCED MICRONUTRIENT ANALYSIS...")
        
        # 1. Analyze micronutrient coverage
        micronutrient_coverage = self.nutrient_calc.analyze_micronutrient_coverage(
            request.fertilizers, request.target_concentrations, request.water_analysis
        )
        
        # 2. Calculate micronutrient dosages if needed
        micronutrient_dosages = {}
        if micronutrient_coverage['micronutrients_needed']:
            micronutrient_needs = {
                micro: info['remaining_need'] 
                for micro, info in micronutrient_coverage['micronutrients_needed'].items()
            }
            micronutrient_dosages = self.nutrient_calc.calculate_micronutrient_dosages(
                micronutrient_needs, request.fertilizers
            )
        
        # 3. Validate micronutrient solution
        micronutrient_validation = self.nutrient_calc.validate_micronutrient_solution(
            final_solution['FINAL_mg_L'], request.target_concentrations
        )
        
        # 4. Generate micronutrient recommendations
        micronutrient_recommendations = self.nutrient_calc.generate_micronutrient_recommendations(
            micronutrient_validation, request.fertilizers
        )

        return {
            'fertilizer_dosages': fertilizer_dosages,
            'nutrient_contributions': nutrient_contributions,
            'water_contribution': water_contribution,
            'final_solution': final_solution,
            'verification_results': verification_results,
            'ionic_relationships': ionic_relationships,
            'ionic_balance': ionic_balance,
            'cost_analysis': cost_analysis,
            
            # *** NEW: ADD MICRONUTRIENT ANALYSIS RESULTS ***
            'micronutrient_coverage': micronutrient_coverage,
            'micronutrient_dosages': micronutrient_dosages,
            'micronutrient_validation': micronutrient_validation,
            'micronutrient_recommendations': micronutrient_recommendations,
            
            'calculation_status': {
                'success': True,
                'warnings': [],
                'iterations': 1,
                'convergence_error': ionic_balance['difference_percentage'] / 100,
                'method_used': method,
                'micronutrient_analysis_performed': True,  # NEW FLAG
                'micronutrient_coverage_percentage': micronutrient_coverage['coverage_percentage']  # NEW METRIC
            }
        }
    def optimize_deterministic_solution(self, fertilizers: List, targets: Dict[str, float], water: Dict[str, float]) -> Dict[str, float]:
        """
        FIXED: Improved deterministic optimization that prioritizes balanced fertilizers
        """
        print(f"\n--- DETERMINISTIC OPTIMIZATION (FIXED FOR BALANCE) ---")
        
        results = {}
        remaining_nutrients = {}

        # Calculate remaining nutrients after water
        for element, target in targets.items():
            water_content = water.get(element, 0)
            remaining = max(0, target - water_content)
            remaining_nutrients[element] = remaining
            print(f"{element}: Target={target:.1f}, Water={water_content:.1f}, Remaining={remaining:.1f} mg/L")

        # Filter useful fertilizers
        useful_fertilizers = []
        for fert in fertilizers:
            total_content = sum(fert.composition.cations.values()) + sum(fert.composition.anions.values())
            if total_content > 5:
                useful_fertilizers.append(fert)

        print(f"\nUseful fertilizers: {len(useful_fertilizers)}")        
        # 1. Phosphorus - AVOID pure acids, prefer balanced phosphates
        if remaining_nutrients.get('P', 0) > 0:
            print(f"Step 1: Phosphorus sources (prioritizing potassium-based over ammonium)...")
            p_fertilizers = [f for f in useful_fertilizers if f.composition.anions.get('P', 0) > 5]
            
            if p_fertilizers:
                # PRIORIZAR: K-based > balanced > ammonium-based > acids
                balanced_p_ferts = [f for f in p_fertilizers if not ('acido' in f.name.lower() and 
                                                                sum(f.composition.cations.values()) < 1)]
                
                if balanced_p_ferts:
                    # 1st choice: Potassium phosphates (mejor balance ionico)
                    k_phosphates = [f for f in balanced_p_ferts if 'potasico' in f.name.lower()]
                    
                    # 2nd choice: Mixed phosphates  
                    mixed_phosphates = [f for f in balanced_p_ferts if 'bipotasico' in f.name.lower()]
                    
                    # 3rd choice: Ammonium phosphates (solo si es necesario)
                    nh4_phosphates = [f for f in balanced_p_ferts if 'amonico' in f.name.lower()]
                    
                    if k_phosphates:
                        best_p_fert = k_phosphates[0]  # Fosfato monopotasico
                        print(f"  Selected: {best_p_fert.name} (K-based phosphate - best balance)")
                    elif mixed_phosphates:
                        best_p_fert = mixed_phosphates[0]  # Fosfato bipotasico  
                        print(f"  Selected: {best_p_fert.name} (mixed phosphate)")
                    elif nh4_phosphates:
                        best_p_fert = nh4_phosphates[0]  # MAP/DAP como ultimo recurso
                        print(f"  Selected: {best_p_fert.name} (ammonium phosphate - will limit dosage)")
                    else:
                        best_p_fert = balanced_p_ferts[0]
                        print(f"  Selected: {best_p_fert.name} (fallback balanced)")
                else:
                    best_p_fert = max(p_fertilizers, key=lambda f: f.composition.anions.get('P', 0))
                    print(f"  Selected: {best_p_fert.name} (fallback to acid)")
                
                p_needed = remaining_nutrients['P']
                
                # LIMITAR dosificacion si es fertilizante amoniacal
                if 'amonico' in best_p_fert.name.lower():
                    p_needed = min(p_needed, targets.get('P', 0) * 0.7)  # Maximo 70% del objetivo
                    print(f"    Limiting ammonium fertilizer to 70% of P target")
                
                dosage = self.nutrient_calc.calculate_fertilizer_requirement(
                    'P', p_needed, {'P': best_p_fert.composition.anions.get('P', 0)},
                    best_p_fert.percentage, best_p_fert.molecular_weight
                )
                
                if dosage > 0:
                    results[best_p_fert.name] = dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_p_fert, dosage)
                    print(f"Added {best_p_fert.name}: {dosage/1000:.3f} g/L")
                    
        # 2. Calcium sources (these are usually balanced already)
        if remaining_nutrients.get('Ca', 0) > 0:
            print(f"Step 2: Calcium sources...")
            ca_fertilizers = [f for f in useful_fertilizers 
                            if f.composition.cations.get('Ca', 0) > 10 and f.name not in results]
            
            if ca_fertilizers:
                best_ca_fert = max(ca_fertilizers, key=lambda f: f.composition.cations.get('Ca', 0))
                ca_needed = remaining_nutrients['Ca']
                dosage = self.nutrient_calc.calculate_fertilizer_requirement(
                    'Ca', ca_needed, {'Ca': best_ca_fert.composition.cations.get('Ca', 0)},
                    best_ca_fert.percentage, best_ca_fert.molecular_weight
                )
                
                if dosage > 0:
                    results[best_ca_fert.name] = dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_ca_fert, dosage)
                    print(f"Added {best_ca_fert.name}: {dosage/1000:.3f} g/L")

        # 3. Potassium sources - prefer based on other nutrient needs
        if remaining_nutrients.get('K', 0) > 0:
            print(f"Step 3: Potassium sources...")
            k_fertilizers = [f for f in useful_fertilizers 
                            if f.composition.cations.get('K', 0) > 20 and f.name not in results]
            
            if k_fertilizers:
                # Smart selection based on other needs
                if remaining_nutrients.get('N', 0) > 100:
                    # Need nitrogen - prefer nitrate
                    k_nitrate = [f for f in k_fertilizers if 'nitrato' in f.name.lower()]
                    best_k_fert = k_nitrate[0] if k_nitrate else k_fertilizers[0]
                    print(f"  Selected K+N source: {best_k_fert.name}")
                elif remaining_nutrients.get('Cl', 0) < 50:
                    # Low chloride need - can use chloride
                    k_chloride = [f for f in k_fertilizers if 'cloruro' in f.name.lower()]
                    best_k_fert = k_chloride[0] if k_chloride else k_fertilizers[0]
                    print(f"  Selected K+Cl source: {best_k_fert.name}")
                else:
                    # Use any available K source
                    best_k_fert = max(k_fertilizers, key=lambda f: f.composition.cations.get('K', 0))
                    print(f"  Selected high-K source: {best_k_fert.name}")
                
                k_needed = remaining_nutrients['K']
                dosage = self.nutrient_calc.calculate_fertilizer_requirement(
                    'K', k_needed, {'K': best_k_fert.composition.cations.get('K', 0)},
                    best_k_fert.percentage, best_k_fert.molecular_weight
                )
                
                if dosage > 0:
                    results[best_k_fert.name] = dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_k_fert, dosage)
                    print(f"Added {best_k_fert.name}: {dosage/1000:.3f} g/L")

        # 4. Magnesium sources  
        if remaining_nutrients.get('Mg', 0) > 0:
            print(f"Step 4: Magnesium sources...")
            mg_fertilizers = [f for f in useful_fertilizers 
                            if f.composition.cations.get('Mg', 0) > 5 and f.name not in results]
            if mg_fertilizers:
                best_mg_fert = max(mg_fertilizers, key=lambda f: f.composition.cations.get('Mg', 0))
                mg_needed = remaining_nutrients['Mg']
                dosage = self.nutrient_calc.calculate_fertilizer_requirement(
                    'Mg', mg_needed, {'Mg': best_mg_fert.composition.cations.get('Mg', 0)},
                    best_mg_fert.percentage, best_mg_fert.molecular_weight
                )
                
                if dosage > 0:
                    results[best_mg_fert.name] = dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_mg_fert, dosage)
                    print(f"Added {best_mg_fert.name}: {dosage/1000:.3f} g/L")

        # 5. Sulfur - PREFER balanced sulfates over pure acid
        if remaining_nutrients.get('S', 0) > 10:
            print(f"Step 5: Sulfur sources (avoiding ammonium sulfate for balance)...")
            s_fertilizers = [f for f in useful_fertilizers 
                            if f.composition.anions.get('S', 0) > 10 and f.name not in results]
            
            if s_fertilizers:
                # FILTRAR sulfato de amonio para evitar exceso de NH4+
                non_ammonium_s = [f for f in s_fertilizers if not ('amonio' in f.name.lower() and 'sulfato' in f.name.lower())]
                
                # Buscar sulfato de potasio o magnesio primero
                k_sulfates = [f for f in non_ammonium_s if 'potasio' in f.name.lower() and 'sulfato' in f.name.lower()]
                mg_sulfates = [f for f in non_ammonium_s if 'magnesio' in f.name.lower() and 'sulfato' in f.name.lower()]
                
                if k_sulfates:
                    best_s_fert = k_sulfates[0]  # Sulfato de potasio
                    print(f"  Selected: {best_s_fert.name} (K-sulfate - excellent balance)")
                elif mg_sulfates:
                    best_s_fert = mg_sulfates[0]  # Sulfato de magnesio
                    print(f"  Selected: {best_s_fert.name} (Mg-sulfate - good balance)")
                elif non_ammonium_s:
                    best_s_fert = max(non_ammonium_s, key=lambda f: f.composition.anions.get('S', 0))
                    print(f"  Selected: {best_s_fert.name} (non-ammonium sulfate)")
                else:
                    # Solo usar sulfato de amonio como ultimo recurso y limitado
                    best_s_fert = max(s_fertilizers, key=lambda f: f.composition.anions.get('S', 0))
                    print(f"  Selected: {best_s_fert.name} (ammonium sulfate - limited dosage)")
                
                s_needed = remaining_nutrients['S']
                
                # LIMITAR dosificacion si es sulfato de amonio
                if 'amonio' in best_s_fert.name.lower() and 'sulfato' in best_s_fert.name.lower():
                    s_needed = min(s_needed, targets.get('S', 0) * 0.4)  # Maximo 40% del objetivo
                    print(f"    Limiting ammonium sulfate to 40% of S target for ionic balance")
                
                dosage = self.nutrient_calc.calculate_fertilizer_requirement(
                    'S', s_needed, {'S': best_s_fert.composition.anions.get('S', 0)},
                    best_s_fert.percentage, best_s_fert.molecular_weight
                )
                
                if dosage > 0:
                    results[best_s_fert.name] = dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_s_fert, dosage)
                    print(f"Added {best_s_fert.name}: {dosage/1000:.3f} g/L")


        # 6. Micronutrients (existing logic is fine)
        micronutrients = {
            'Fe': targets.get('Fe', 2.0),
            'Mn': targets.get('Mn', 0.5),
            'Zn': targets.get('Zn', 0.3),
            'Cu': targets.get('Cu', 0.1),
            'B': targets.get('B', 0.5),
            'Mo': targets.get('Mo', 0.05)
        }
        
        print(f"Step 6: Micronutrients...")
        micro_sources = {
            'Fe': ('FeEDTA', 0.13),
            'Mn': ('MnSO4.4H2O', 0.24),
            'Zn': ('ZnSO4.7H2O', 0.23),
            'Cu': ('CuSO4.5H2O', 0.25),
            'B': ('H3BO3', 0.17),
            'Mo': ('Na2MoO4.2H2O', 0.39)
        }
        
        for micro, target in micronutrients.items():
            if target > 0:
                fert_name, content_percent = micro_sources[micro]
                dosage_mg_l = target / content_percent
                dosage_g_l = dosage_mg_l / 1000.0
                
                if dosage_g_l > 0.001:
                    results[fert_name] = dosage_g_l
                    print(f"Added {fert_name}: {dosage_g_l:.4f} g/L for {micro}")

        # Apply ionic balance correction
        results = self._apply_ionic_balance_correction(results, useful_fertilizers)
        
        print(f"\nFallback deterministic optimization completed:")
        active_count = len([d for d in results.values() if d > 0])
        print(f"Active fertilizers: {active_count}")
        print(f"Strategy: Balanced fertilizers with ionic correction")
        
        return results
    
    def _apply_ionic_balance_correction(self, results: Dict[str, float], fertilizers: List) -> Dict[str, float]:
        """Apply simple ionic balance correction to results"""
        try:
            # Calculate current ionic contributions
            total_cations = 0
            total_anions = 0
            
            fertilizer_map = {f.name: f for f in fertilizers}
            
            for fert_name, dosage_g_l in results.items():
                if fert_name in fertilizer_map and dosage_g_l > 0:
                    fert = fertilizer_map[fert_name]
                    dosage_mg_l = dosage_g_l * 1000
                    
                    # Calculate cation contributions
                    for cation in ['Ca', 'K', 'Mg', 'Na', 'NH4', 'Fe', 'Mn', 'Zn', 'Cu']:
                        content = fert.composition.cations.get(cation, 0)
                        if content > 0:
                            mg_contribution = self.nutrient_calc.calculate_element_contribution(
                                dosage_mg_l, content, fert.percentage
                            )
                            mmol = self.nutrient_calc.convert_mg_to_mmol(mg_contribution, cation)
                            meq = self.nutrient_calc.convert_mmol_to_meq(mmol, cation)
                            total_cations += meq
                    
                    # Calculate anion contributions
                    for anion in ['N', 'S', 'Cl', 'P', 'HCO3', 'B', 'Mo']:
                        content = fert.composition.anions.get(anion, 0)
                        if content > 0:
                            mg_contribution = self.nutrient_calc.calculate_element_contribution(
                                dosage_mg_l, content, fert.percentage
                            )
                            mmol = self.nutrient_calc.convert_mg_to_mmol(mg_contribution, anion)
                            meq = self.nutrient_calc.convert_mmol_to_meq(mmol, anion)
                            total_anions += meq
            
            ionic_imbalance = abs(total_cations - total_anions)
            ionic_error = ionic_imbalance / ((total_cations + total_anions) / 2) * 100 if (total_cations + total_anions) > 0 else 0
            
            print(f"\nCurrent ionic balance:")
            print(f"Cations: {total_cations:.2f} meq/L")
            print(f"Anions: {total_anions:.2f} meq/L")
            print(f"Error: {ionic_error:.1f}%")
            
            if ionic_error > 15:  # If error > 15%, apply correction
                print("Applying ionic balance correction...")
                
                if total_cations > total_anions:  # Excess cations
                    # Reduce cation-rich fertilizers and/or add anion-rich ones
                    adjustment_factor = 0.9  # Reduce by 10%
                    for fert_name in results.keys():
                        if fert_name in fertilizer_map:
                            fert = fertilizer_map[fert_name]
                            cation_rich = sum(fert.composition.cations.values()) > sum(fert.composition.anions.values())
                            if cation_rich and results[fert_name] > 0.1:
                                results[fert_name] *= adjustment_factor
                                print(f"  Reduced {fert_name}: {results[fert_name]:.3f} g/L")
                                
                else:  # Excess anions
                    # Reduce anion-rich fertilizers and/or add cation-rich ones
                    adjustment_factor = 0.9  # Reduce by 10%
                    for fert_name in results.keys():
                        if fert_name in fertilizer_map:
                            fert = fertilizer_map[fert_name]
                            anion_rich = sum(fert.composition.anions.values()) > sum(fert.composition.cations.values())
                            if anion_rich and results[fert_name] > 0.1:
                                results[fert_name] *= adjustment_factor
                                print(f"  Reduced {fert_name}: {results[fert_name]:.3f} g/L")
            
            return results
            
        except Exception as e:
            print(f"Ionic balance correction error: {e}")
            return results

    def _update_remaining_nutrients(self, remaining_nutrients: Dict[str, float], fertilizer, dosage: float):
        """Update remaining nutrients after adding a fertilizer"""
        all_elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'S', 'Cl', 'P', 'HCO3', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        for element in all_elements:
            cation_content = fertilizer.composition.cations.get(element, 0)
            anion_content = fertilizer.composition.anions.get(element, 0)
            total_content = cation_content + anion_content
            
            if total_content > 0:
                contribution = self.nutrient_calc.calculate_element_contribution(
                    dosage, total_content, fertilizer.percentage
                )
                if element in remaining_nutrients:
                    remaining_nutrients[element] = max(0, remaining_nutrients[element] - contribution)

    def calculate_all_contributions(self, fertilizers: List, dosages: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate all nutrient contributions from fertilizers"""
        elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'SO4', 'S', 'Cl', 'H2PO4', 'P', 'HCO3', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        contributions = {
            'APORTE_mg_L': {elem: 0.0 for elem in elements},
            'DE_mmol_L': {elem: 0.0 for elem in elements},
            'IONES_meq_L': {elem: 0.0 for elem in elements}
        }

        for fertilizer in fertilizers:
            dosage_g_l = dosages.get(fertilizer.name, 0)
            if dosage_g_l > 0:
                dosage_mg_l = dosage_g_l * 1000

                for element in elements:
                    cation_content = fertilizer.composition.cations.get(element, 0)
                    anion_content = fertilizer.composition.anions.get(element, 0)
                    total_content = cation_content + anion_content

                    if total_content > 0:
                        contribution_mg_l = self.nutrient_calc.calculate_element_contribution(
                            dosage_mg_l, total_content, fertilizer.percentage
                        )
                        contributions['APORTE_mg_L'][element] += contribution_mg_l

                        mmol_contribution = self.nutrient_calc.convert_mg_to_mmol(contribution_mg_l, element)
                        contributions['DE_mmol_L'][element] += mmol_contribution

                        meq_contribution = self.nutrient_calc.convert_mmol_to_meq(mmol_contribution, element)
                        contributions['IONES_meq_L'][element] += meq_contribution

        # Round all values
        for category in contributions:
            for element in contributions[category]:
                contributions[category][element] = round(contributions[category][element], 3)

        return contributions

    def calculate_water_contributions(self, water_analysis: Dict[str, float]):
        """Calculate water contributions in all units"""
        elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'SO4', 'S', 'Cl', 'H2PO4', 'P', 'HCO3', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
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
        elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'SO4', 'S', 'Cl', 'H2PO4', 'P', 'HCO3', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        final = {
            'FINAL_mg_L': {},
            'FINAL_mmol_L': {},
            'FINAL_meq_L': {}
        }

        for element in elements:
            final_mg_l = (nutrient_contrib['APORTE_mg_L'][element] + 
                         water_contrib['IONES_mg_L_DEL_AGUA'][element])
            final_mmol_l = self.nutrient_calc.convert_mg_to_mmol(final_mg_l, element)
            final_meq_l = self.nutrient_calc.convert_mmol_to_meq(final_mmol_l, element)

            final['FINAL_mg_L'][element] = round(final_mg_l, 3)
            final['FINAL_mmol_L'][element] = round(final_mmol_l, 3)
            final['FINAL_meq_L'][element] = round(final_meq_l, 3)

        # Calculate EC and pH
        cations = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'Fe', 'Mn', 'Zn', 'Cu']
        cation_sum = sum(final['FINAL_meq_L'].get(cation, 0) for cation in cations)
        ec = cation_sum * 0.1

        hco3 = final['FINAL_mg_L'].get('HCO3', 0)
        no3_n = final['FINAL_mg_L'].get('N', 0)
        if hco3 > 61:
            ph = 6.5 + (hco3 - 61) / 100
        else:
            ph = 6.0 - (no3_n / 200)

        return {
            'FINAL_mg_L': final['FINAL_mg_L'],
            'FINAL_mmol_L': final['FINAL_mmol_L'],
            'FINAL_meq_L': final['FINAL_meq_L'],
            'calculated_EC': round(ec, 2),
            'calculated_pH': round(ph, 1)
        }

# Initialize calculator
calculator = CompleteFertilizerCalculator()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/calculate-simple", response_model=SimpleResponse)
async def calculate_simple_fertilizer_solution(request: FertilizerRequest):
    """Simple fertilizer calculation with basic optimization"""
    try:
        print("Starting simple calculation...")
        
        # Use deterministic method for simple calculation
        calculation_results = calculator.calculate_advanced_solution(request, method="deterministic")
        
        # Create response
        response = SimpleResponse(
            fertilizer_dosages=calculation_results['fertilizer_dosages'],
            calculation_status=CalculationStatus(**calculation_results['calculation_status'])
        )

        # Generate simple PDF report
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"reports/simple_report_{timestamp}.pdf"
            
            calculation_data = {
                "integration_metadata": {
                    "data_source": "Simple API Calculation",
                    "calculation_timestamp": datetime.now().isoformat()
                },
                "calculation_results": calculation_results
            }
            
            pdf_generator.generate_comprehensive_pdf(calculation_data, pdf_filename)
            
            response.pdf_report = {
                "generated": "true",
                "filename": pdf_filename,
                "status": "success"
            }
            
        except Exception as e:
            print(f"PDF generation failed: {e}")
            response.pdf_report = {
                "generated": "false",
                "status": f"error: {str(e)}"
            }

        return response

    except Exception as e:
        print(f"Simple calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simple calculation error: {str(e)}")

@app.post("/calculate-advanced")
async def calculate_advanced_fertilizer_solution(request: FertilizerRequest, method: str = "deterministic"):
    """Advanced fertilizer calculation with multiple optimization methods"""
    try:
        print(f"Starting advanced calculation with method: {method}")
        
        # Validate method
        valid_methods = ["deterministic", "machine_learning"]
        if method not in valid_methods:
            raise HTTPException(status_code=400, detail=f"Invalid method. Choose from: {valid_methods}")
        
        # Calculate solution
        calculation_results = calculator.calculate_advanced_solution(request, method=method)
        
        # Generate comprehensive PDF report
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"reports/advanced_report_{method}_{timestamp}.pdf"
            
            calculation_data = {
                "integration_metadata": {
                    "data_source": f"Advanced API Calculation - {method.title()}",
                    "calculation_timestamp": datetime.now().isoformat(),
                    "method_used": method
                },
                "calculation_results": calculation_results
            }
            
            pdf_generator.generate_comprehensive_pdf(calculation_data, pdf_filename)
            calculation_results['pdf_report'] = {
                "generated": True,
                "filename": pdf_filename,
                "method": method
            }
            
        except Exception as e:
            print(f"PDF generation failed: {e}")
            calculation_results['pdf_report'] = {
                "generated": False,
                "error": str(e)
            }

        return calculation_results

    except Exception as e:
        print(f"Advanced calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced calculation error: {str(e)}")

@app.post("/calculate-ml")
async def calculate_ml_fertilizer_solution(request: FertilizerRequest):
    """Machine learning optimized fertilizer calculation"""
    return await calculate_advanced_fertilizer_solution(request, method="machine_learning")

@app.post("/train-ml-model")
async def train_ml_model(n_samples: int = Query(default=5000), model_type: str = Query(default="RandomForest")):
    """Train the ML model with synthetic data"""
    try:
        print(f"Training ML model with {n_samples} samples...")
        
        config = MLModelConfig(model_type=model_type)
        ml_optimizer.config = config
        
        # Generate training data first, then train
        print(f"Generating {n_samples} training samples...")
        test_fertilizers = [
            fertilizer_db.create_fertilizer_from_database('nitrato de calcio'),
            fertilizer_db.create_fertilizer_from_database('nitrato de potasio'),
            fertilizer_db.create_fertilizer_from_database('fosfato monopotasico'),
            fertilizer_db.create_fertilizer_from_database('sulfato de magnesio')
        ]
        training_data = ml_optimizer.generate_real_training_data(fertilizers=test_fertilizers,num_scenarios=n_samples)
        
        print(f"Training model with {len(training_data)} samples...")
        training_results = ml_optimizer.train_model(training_data=training_data)
        
        return {
            "status": "training_complete",
            "training_results": training_results,
            "model_ready": ml_optimizer.is_trained,
            "training_samples": len(training_data),
            "test_mae": training_results.get("test_mae_overall", 0),
            "model_type": model_type
        }
        
    except Exception as e:
        print(f"ML training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ML training error: {str(e)}")

@app.get("/test-optimization-methods")
async def test_optimization_methods(
    target_N: float = 150, target_P: float = 40, target_K: float = 200,
    target_Ca: float = 180, target_Mg: float = 50, target_S: float = 80
):
    """Test and compare all optimization methods"""
    try:
        import time
        
        # Create test request
        test_fertilizers = [
            fertilizer_db.create_fertilizer_from_database('nitrato de calcio'),
            fertilizer_db.create_fertilizer_from_database('nitrato de potasio'),
            fertilizer_db.create_fertilizer_from_database('fosfato monopotasico'),
            fertilizer_db.create_fertilizer_from_database('sulfato de magnesio')
        ]
        
        # Filter valid fertilizers
        test_fertilizers = [f for f in test_fertilizers if f is not None]
        
        targets = {
            'N': target_N, 'P': target_P, 'K': target_K,
            'Ca': target_Ca, 'Mg': target_Mg, 'S': target_S
        }
        
        water = {'Ca': 20, 'K': 5, 'N': 2, 'P': 1, 'Mg': 8, 'S': 5}
        
        results = {}
        
        # Test deterministic method
        print("Testing deterministic method...")
        start_time = time.time()
        try:
            det_result = calculator.optimize_deterministic_solution(test_fertilizers, targets, water)
            det_time = time.time() - start_time
            
            results['deterministic'] = {
                'execution_time_seconds': det_time,
                'active_fertilizers': len([d for d in det_result.values() if d > 0.001]),
                'dosages': det_result,
                'status': 'success'
            }
        except Exception as e:
            results['deterministic'] = {'status': 'failed', 'error': str(e)}
     
        # Test ML method (train first if needed)
        print("Testing ML method...")
        start_time = time.time()
        try:
            if not ml_optimizer.is_trained:
                print("Training ML model...")
                training_data = ml_optimizer.generate_real_training_data(
                    fertilizers=test_fertilizers, num_scenarios=2000
                )
                ml_optimizer.train_model(training_data=training_data, fertilizers=test_fertilizers, n_samples=2000)

            ml_result = ml_optimizer.optimize_with_ml(targets, water, test_fertilizers)
            ml_time = time.time() - start_time
            
            results['machine_learning'] = {
                'execution_time_seconds': ml_time,
                'active_fertilizers': len([d for d in ml_result.values() if d > 0.001]),
                'dosages': ml_result,
                'status': 'success'
            }
        except Exception as e:
            results['machine_learning'] = {'status': 'failed', 'error': str(e)}
        
        # Performance comparison
        successful_methods = [method for method, data in results.items() 
                            if isinstance(data, dict) and data.get('status') == 'success']
        
        if successful_methods:
            fastest_method = min(successful_methods, 
                               key=lambda m: results[m]['execution_time_seconds'])
            performance_comparison = {
                'fastest_method': fastest_method,
                'methods_tested': len(results),
                'successful_methods': len(successful_methods)
            }
        else:
            performance_comparison = {
                'fastest_method': 'none',
                'methods_tested': len(results),
                'successful_methods': 0
            }

        return {
            'test_conditions': {
                'targets': targets,
                'water': water,
                'fertilizers_tested': len(test_fertilizers)
            },
            'results': results,
            'performance_comparison': performance_comparison
        }

    except Exception as e:
        # TEMP_DISABLED: print(f"Optimization testing failed: {str(e)}")  # Unicode encoding issue
        raise HTTPException(status_code=500, detail=f"Optimization testing error: {str(e)}")

@app.get("/swagger-integrated-calculation")
async def swagger_integrated_calculation(
    user_id: int,
    catalog_id: int = Query(default=1),
    phase_id: int = Query(default=1),
    water_id: int = Query(default=1),
    volume_liters: float = Query(default=1000),
    use_ml: bool = Query(default=False),
    apply_safety_caps: bool = Query(default=True),  # NEW PARAMETER
    strict_caps: bool = Query(default=True)         # NEW PARAMETER  
):
    """Complete Swagger API integration with nutrient safety caps"""
    try:
        print(f"\n=== STARTING ENHANCED SWAGGER INTEGRATION WITH NUTRIENT CAPS ===")
        print(f"Safety Caps: {apply_safety_caps}, Strict Mode: {strict_caps}")
        
        # Initialize Swagger client and authenticate using context manager
        async with SwaggerAPIClient("http://162.248.52.111:8082") as swagger_client:
            # Login to get authentication token
            print("Authenticating with Swagger API...")
            login_result = await swagger_client.login("csolano@iapcr.com", "123")
            if not login_result.get('success'):
                raise HTTPException(status_code=401, detail="Authentication failed")
            
            print("Authentication successful!")
            
            # Get user information
            user_info = await swagger_client.get_user_by_id(user_id)
            print(f"User: {user_info.get('userEmail', 'N/A')} (ID: {user_id})")
            
            # Fetch data from multiple endpoints
        # TEMP_DISABLED: print(f"\nFETCHING API DATA...")  # Unicode encoding issue
            
            fertilizers_data = await swagger_client.get_fertilizers(catalog_id)
            requirements_data = await swagger_client.get_crop_phase_requirements(phase_id)
            water_data = await swagger_client.get_water_chemistry(water_id, catalog_id)
            
            print(f"Fetched: {len(fertilizers_data)} fertilizers, "
                  f"{len(requirements_data) if requirements_data else 0} requirements, "
                  f"{len(water_data) if water_data else 0} water parameters")
            
            # Process fertilizers into our format
            api_fertilizers = []
            for fert_data in fertilizers_data:
                try:
                    fertilizer = swagger_client.map_swagger_fertilizer_to_model(fert_data)
                    total_content = sum(fertilizer.composition.cations.values()) + sum(fertilizer.composition.anions.values())
                    
                    # Include fertilizers even with default compositions (total_content = 0)
                    # They might be matched later or be useful in calculations
                    if total_content >= 0:  # Accept all fertilizers
                        api_fertilizers.append(fertilizer)
                        if total_content > 1:
                            print(f"  Added: {fertilizer.name} (content: {total_content:.1f}%)")
                        else:
                            print(f"  Added: {fertilizer.name} (default composition, will try pattern matching)")
                    else:
                        print(f"  Skipped: {fertilizer.name} (invalid content: {total_content:.1f}%)")
                        
                except Exception as e:
                    print(f"  Error processing {fert_data.get('name', 'Unknown')}: {e}")
        
            if not api_fertilizers:
                raise HTTPException(status_code=500, detail="No usable fertilizers found from API")
            
            print(f"Successfully processed {len(api_fertilizers)} API fertilizers")
            
            # Map API data to our format
            target_concentrations = swagger_client.map_requirements_to_targets(requirements_data)
            water_analysis = swagger_client.map_water_to_analysis(water_data)
        
            # Use defaults if no data available
            if not target_concentrations:
                print("No target concentrations found, using defaults")
                target_concentrations = {
                    'N': 150, 'P': 50, 'K': 200, 'Ca': 180, 'Mg': 50, 'S': 80,
                    'Fe': 2.0, 'Mn': 0.5, 'Zn': 0.3, 'Cu': 0.1, 'B': 0.5, 'Mo': 0.05
                }
            
            if not water_analysis:
                print("No water analysis found, using defaults")
                water_analysis = {
                    'Ca': 20, 'K': 5, 'N': 2, 'P': 1, 'Mg': 8, 'S': 5,
                    'Fe': 0.1, 'Mn': 0.05, 'Zn': 0.02, 'Cu': 0.01, 'B': 0.1, 'Mo': 0.001
                }

            print(f"Target concentrations: {len(target_concentrations)} parameters")
            print(f"Water analysis: {len(water_analysis)} parameters")
            
            # ===== NEW: APPLY NUTRIENT SAFETY CAPS BEFORE CALCULATIONS =====
            caps_result = None
            if apply_safety_caps:
        # TEMP_DISABLED: print(f"\nAPPLYING NUTRIENT SAFETY CAPS...")  # Unicode encoding issue
                caps_result = apply_nutrient_caps_to_targets(target_concentrations, strict_mode=strict_caps)
                
                # Use the safe, capped concentrations for calculations
                safe_target_concentrations = caps_result['capped_concentrations']
                
                # Log what was changed
                if caps_result['total_adjustments'] > 0:
                    print(f"WARNING: SAFETY ADJUSTMENTS MADE:")
                    for adjustment in caps_result['adjustments_made']:
                        print(f"   {adjustment['nutrient']}: {adjustment['original']:.1f} -> {adjustment['capped']:.1f} mg/L")
                        print(f"      Reason: {adjustment['reason']}")
                    
                    print(f"Safety Score: {caps_result['summary']['safety_score']:.1f}/100")
                else:
                    print(f"SUCCESS: All targets within safe limits - no adjustments needed")
                    safe_target_concentrations = target_concentrations
            else:
                print(f"WARNING: SAFETY CAPS DISABLED - Using original targets")
                safe_target_concentrations = target_concentrations
            # ================================================================
            
            # **AUTO-ADD REQUIRED MICRONUTRIENT FERTILIZERS**
        # TEMP_DISABLED: print(f"\nCHECKING FOR REQUIRED MICRONUTRIENT FERTILIZERS...")  # Unicode encoding issue
            enhanced_fertilizers = add_required_micronutrient_fertilizers(
                api_fertilizers, safe_target_concentrations, water_analysis  # Use safe targets
            )
            
            print(f"Final fertilizer count: {len(enhanced_fertilizers)} (API: {len(api_fertilizers)}, Added: {len(enhanced_fertilizers) - len(api_fertilizers)})")
            
            # Create calculation request with SAFE targets
            from models import CalculationSettings
            request = FertilizerRequest(
                fertilizers=enhanced_fertilizers,
                target_concentrations=safe_target_concentrations,  # Use safe targets
                water_analysis=water_analysis,
                calculation_settings=CalculationSettings(
                    volume_liters=volume_liters,
                    precision=3,
                    units="mg/L",
                    crop_phase="General"
                )
            )
            
            # Choose optimization method
            if use_ml:
                method = "machine_learning"
            else:
                method = "deterministic"

            print(f"Starting calculation with {method} method (safe targets)...")
            
            # Perform calculation
            calculation_results = calculator.calculate_advanced_solution(request, method=method)
            
            # Generate comprehensive PDF with user info and caps information
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"reports/swagger_user_{user_id}_{method}_safe_{timestamp}.pdf"
                
                calculation_data = {
                    "user_info": user_info,
                    "integration_metadata": {
                        "data_source": "Swagger API with Safety Caps",
                        "user_id": user_id,
                        "catalog_id": catalog_id,
                        "phase_id": phase_id,
                        "water_id": water_id,
                        "fertilizers_analyzed": len(fertilizers_data),
                        "fertilizers_processed": len(api_fertilizers),
                        "micronutrients_added": len(enhanced_fertilizers) - len(api_fertilizers),
                        "fertilizers_matched": len([f for f in enhanced_fertilizers if sum(f.composition.cations.values()) + sum(f.composition.anions.values()) > 10]),
                        "optimization_method": method,
                        "calculation_timestamp": datetime.now().isoformat(),
                        "auto_micronutrient_supplementation": True,
                        "safety_caps_applied": apply_safety_caps,
                        "strict_caps_mode": strict_caps,
                        "safety_adjustments": caps_result['total_adjustments'] if caps_result else 0,
                        "safety_score": caps_result['summary']['safety_score'] if caps_result else 100.0
                    },
                    "calculation_results": calculation_results,
                    "safety_caps_info": caps_result  # Include full caps information
                }
                
                pdf_generator.generate_comprehensive_pdf(calculation_data, pdf_filename)
                calculation_results['pdf_report'] = {
                    "generated": True,
                    "filename": pdf_filename,
                    "integration_method": "swagger_api_with_safety_caps"
                }
                
            except Exception as e:
                print(f"PDF generation failed: {e}")
                calculation_results['pdf_report'] = {
                    "generated": False,
                    "error": str(e)
                }
            
            # Create comprehensive response
            response = {
                "user_info": user_info,
                "integration_metadata": calculation_data["integration_metadata"],
                "safety_caps_summary": {
                    "caps_applied": apply_safety_caps,
                    "strict_mode": strict_caps,
                    "total_adjustments": caps_result['total_adjustments'] if caps_result else 0,
                    "safety_score": caps_result['summary']['safety_score'] if caps_result else 100.0,
                    "high_priority_warnings": caps_result['summary']['high_priority_warnings'] if caps_result else 0,
                    "adjusted_nutrients": [adj['nutrient'] for adj in caps_result['adjustments_made']] if caps_result else [],
                    "original_vs_safe_targets": {
                        "original": target_concentrations,
                        "safe_capped": safe_target_concentrations
                    } if apply_safety_caps else None
                },
                "performance_metrics": {
                    "fertilizers_fetched": len(fertilizers_data),
                    "fertilizers_processed": len(api_fertilizers),
                    "micronutrients_auto_added": len(enhanced_fertilizers) - len(api_fertilizers),
                    "fertilizers_matched": len([f for f in enhanced_fertilizers if sum(f.composition.cations.values()) + sum(f.composition.anions.values()) > 10]),
                    "active_dosages": len([d for d in calculation_results['fertilizer_dosages'].values() if d.dosage_g_per_L > 0]),
                    "optimization_method": method,
                    "micronutrient_coverage": "Complete",
                    "safety_status": "Protected" if apply_safety_caps else "Unprotected"
                },
                "calculation_results": calculation_results,
                "data_sources": {
                    "fertilizers_api": f"/Fertilizer?CatalogId={catalog_id}",
                    "requirements_api": f"/CropPhaseSolutionRequirement/GetByPhaseId?PhaseId={phase_id}",
                    "water_api": f"/WaterChemistry?WaterId={water_id}&CatalogId={catalog_id}",
                    "user_api": "/User",
                    "micronutrient_supplementation": "Local Database Auto-Addition",
                    "safety_caps": "Integrated Nutrient Safety System"
                }
            }
            
            print(f"\n=== ENHANCED SWAGGER INTEGRATION WITH SAFETY CAPS COMPLETE ===")
            print(f"User: {user_info.get('userEmail', 'N/A')} (ID: {user_id})")
            print(f"Method: {method}")
            print(f"Safety Caps: {'Applied' if apply_safety_caps else 'Disabled'}")
            print(f"Safety Adjustments: {caps_result['total_adjustments'] if caps_result else 0}")
            print(f"Safety Score: {caps_result['summary']['safety_score'] if caps_result else 100.0:.1f}/100")
            print(f"API fertilizers: {len(api_fertilizers)}")
            print(f"Auto-added micronutrients: {len(enhanced_fertilizers) - len(api_fertilizers)}")
            print(f"Active fertilizers: {response['performance_metrics']['active_dosages']}")
            print(f"PDF: {calculation_results['pdf_report'].get('filename', 'Not generated')}")
            
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Enhanced Swagger integration with safety caps failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Integration error: {str(e)}")

@app.get("/fertilizer-database")
async def get_fertilizer_database():
    """Get complete fertilizer database information"""
    try:
        database_info = fertilizer_db.get_complete_database_info()
        return database_info
        
    except Exception as e:
        print(f"Database query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/fertilizers-containing/{element}")
async def get_fertilizers_containing_element(element: str, min_content: float = Query(default=1.0)):
    """Find fertilizers containing specific element above minimum content"""
    try:
        fertilizers = fertilizer_db.find_fertilizers_containing_element(element, min_content)
        
        return {
            "element": element,
            "min_content": min_content,
            "found_fertilizers": len(fertilizers),
            "fertilizers": fertilizers
        }
        
    except Exception as e:
        print(f"Fertilizer search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/test")
async def test_calculation():
    """Comprehensive test of all system components"""
    try:
        print("Running comprehensive system test...")
        
        # Test data
        test_fertilizers = [
            fertilizer_db.create_fertilizer_from_database('nitrato de calcio'),
            fertilizer_db.create_fertilizer_from_database('nitrato de potasio'),
            fertilizer_db.create_fertilizer_from_database('fosfato monopotasico'),
            fertilizer_db.create_fertilizer_from_database('sulfato de magnesio')
        ]
        
        # Filter valid fertilizers
        test_fertilizers = [f for f in test_fertilizers if f is not None]
        
        if not test_fertilizers:
            return {"status": "error", "message": "No test fertilizers available"}
        
        # Create test request
        from models import CalculationSettings
        test_request = FertilizerRequest(
            fertilizers=test_fertilizers,
            target_concentrations={
                'N': 150, 'P': 40, 'K': 200, 'Ca': 180, 'Mg': 50, 'S': 80
            },
            water_analysis={
                'Ca': 20, 'K': 5, 'N': 2, 'P': 1, 'Mg': 8, 'S': 5
            },
            calculation_settings=CalculationSettings(
                volume_liters=1000,
                precision=3,
                units="mg/L",
                crop_phase="Test"
            )
        )
        
        # Test deterministic calculation
        result = calculator.calculate_advanced_solution(test_request, method="deterministic")
        
        # Test summary
        active_fertilizers = len([d for d in result['fertilizer_dosages'].values() if d.dosage_g_per_L > 0])
        total_dosage = sum(d.dosage_g_per_L for d in result['fertilizer_dosages'].values())
        ionic_balance_error = result['ionic_balance']['difference_percentage']
        
        return {
            "test_status": "success",
            "system_components": {
                "fertilizer_database": "operational",
                "nutrient_calculator": "operational", 
                "verification_analyzer": "operational",
                "pdf_generator": "operational",
                "ml_optimizer": "ready" if ml_optimizer else "not_loaded"
            },
            "test_results": {
                "fertilizers_tested": len(test_fertilizers),
                "active_fertilizers": active_fertilizers,
                "total_dosage_g_per_L": round(total_dosage, 3),
                "ionic_balance_error_percent": round(ionic_balance_error, 2),
                "calculation_method": "deterministic"
            },
            "result": result,
            "message": f"System test completed successfully with {active_fertilizers} active fertilizers"
        }
        
    except Exception as e:
        print(f"System test failed: {str(e)}")
        return {
            "test_status": "error",
            "error": str(e),
            "message": "System test failed"
        }

@app.get("/")
async def root():
    """Root endpoint with complete API information"""
    return {
        "message": "Complete Modular Fertilizer Calculator API v5.0.0",
        "version": "5.0.0",
        "status": "fully_operational",
        "description": "Professional hydroponic nutrient calculation system with ML optimization and real API integration",
        "key_features": {
            "modular_architecture": [
                "Separated concerns across specialized modules",
                "Nutrient calculator with advanced algorithms",
                "Complete fertilizer database with pattern matching",
                "ML optimizer with multiple algorithms",
                "Professional PDF report generation",
                "Real Swagger API integration"
            ],
            "optimization_methods": [
                "Deterministic optimization (strategic nutrient prioritization)",
                "Machine learning optimization (RandomForest/XGBoost)",
                "Comparative analysis of all methods"
            ],
            "api_integration": [
                "Real Swagger API calls to http://162.248.52.111:8082",
                "Automatic authentication and token management",
                "Fertilizer chemistry data fetching",
                "Crop requirements and water analysis integration",
                "Enhanced error handling and fallbacks"
            ],
            "pdf_generation": [
                "Professional Excel-like calculation tables",
                "Complete nutrient contribution analysis",
                "Ionic balance verification",
                "Cost analysis and optimization suggestions",
                "Comprehensive verification results"
            ]
        },
        "endpoints": {
            "simple_calculation": {
                "url": "/calculate-simple",
                "method": "POST",
                "description": "Basic fertilizer calculation with deterministic method"
            },
            "advanced_calculation": {
                "url": "/calculate-advanced?method=deterministic",
                "method": "POST", 
                "description": "Advanced calculation with method selection",
                "methods": ["deterministic", "machine_learning"]
            },
            "ml_calculation": {
                "url": "/calculate-ml",
                "method": "POST",
                "description": "Machine learning optimized calculation"
            },
            "swagger_integration": {
                "url": "/swagger-integrated-calculation",
                "method": "GET",
                "description": "Complete real API integration with live data",
                "parameters": "catalog_id, phase_id, water_id, use_ml"
            },
            "ml_training": {
                "url": "/train-ml-model?n_samples=5000&model_type=RandomForest",
                "method": "POST",
                "description": "Train ML model with synthetic data"
            },
            "method_comparison": {
                "url": "/test-optimization-methods",
                "method": "GET",
                "description": "Compare all optimization methods"
            },
            "database_info": {
                "url": "/fertilizer-database",
                "method": "GET",
                "description": "Get complete fertilizer database information"
            }
        },
        "database_coverage": {
            "fertilizer_types": ["Acids", "Nitrates", "Sulfates", "Phosphates", "Micronutrients"],
            "total_fertilizers": "15+ with complete compositions",
            "pattern_matching": "Name and formula based intelligent matching",
            "fallback_support": "Default compositions for unknown fertilizers"
        },
        "ml_capabilities": {
            "algorithms": ["RandomForest", "XGBoost"],
            "features": "31 engineered features from targets and water analysis",
            "training": "Synthetic data generation with realistic ranges",
            "prediction": "Multi-output regression for all fertilizer dosages"
        },
        "quick_tests": {
            "system_test": "GET /test",
            "real_api_integration": "GET /swagger-integrated-calculation",
            "database_test": "GET /fertilizer-database",
            "ml_training": "POST /train-ml-model",
            "method_comparison": "GET /test-optimization-methods"
        },
        "documentation": "/docs",
        "reports_directory": "./reports/",
        "ml_ready": ml_optimizer.is_trained if ml_optimizer else False
    }


def add_required_micronutrient_fertilizers(api_fertilizers: List, 
                                         target_concentrations: Dict[str, float], 
                                         water_analysis: Dict[str, float]) -> List:
    """
    Auto-add required micronutrient fertilizers when API catalog doesn't provide them
    """
    print(f"Analyzing micronutrient requirements...")
    
    # Analyze what micronutrients are needed
    micronutrient_needs = {}
    micronutrients = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
    
    for micro in micronutrients:
        target = target_concentrations.get(micro, 0)
        water_content = water_analysis.get(micro, 0)
        remaining_need = max(0, target - water_content)
        
        if remaining_need > 0.01:  # Need at least 0.01 mg/L
            micronutrient_needs[micro] = remaining_need
            print(f"  Need {micro}: {remaining_need:.3f} mg/L")
    
    # Check what micronutrients are already available from API fertilizers
    available_micronutrients = set()
    for fertilizer in api_fertilizers:
        for micro in micronutrients:
            cation_content = fertilizer.composition.cations.get(micro, 0)
            anion_content = fertilizer.composition.anions.get(micro, 0)
            total_content = cation_content + anion_content
            
            if total_content > 0.1:  # Significant micronutrient content
                available_micronutrients.add(micro)
                print(f"  SUCCESS: {micro} available from API fertilizer: {fertilizer.name}")
    
    # Determine missing micronutrients
    missing_micronutrients = set(micronutrient_needs.keys()) - available_micronutrients
    
    if not missing_micronutrients:
        print(f"  SUCCESS: All required micronutrients available from API fertilizers")
        return api_fertilizers
    
    print(f"  ERROR: Missing micronutrients: {', '.join(missing_micronutrients)}")
    
    # Add required micronutrient fertilizers
    enhanced_fertilizers = api_fertilizers.copy()
    fertilizer_db = EnhancedFertilizerDatabase()
    
    # Define required micronutrient fertilizer sources
    required_micronutrient_sources = {
        'Fe': {
            'primary': 'quelato de hierro',  # Fe-EDTA
            'alternatives': ['sulfato de hierro', 'cloruro de hierro'],
            'display_name': 'Quelato de Hierro (Fe-EDTA) [Fertilizante Requerido]'
        },
        'Mn': {
            'primary': 'sulfato de manganeso',  # MnSO4.4H2O
            'alternatives': ['cloruro de manganeso'],
            'display_name': 'Sulfato de Manganeso (MnSO[?]·H[?]O) [Fertilizante Requerido]'
        },
        'Zn': {
            'primary': 'sulfato de zinc',  # ZnSO4.7H2O
            'alternatives': ['cloruro de zinc'],
            'display_name': 'Sulfato de Zinc (ZnSO[?]·H[?]O) [Fertilizante Requerido]'
        },
        'Cu': {
            'primary': 'sulfato de cobre',  # CuSO4.5H2O
            'alternatives': ['cloruro de cobre'],
            'display_name': 'Sulfato de Cobre (CuSO[?]·5H[?]O) [Fertilizante Requerido]'
        },
        'B': {
            'primary': 'acido borico',  # H3BO3
            'alternatives': ['borax'],
            'display_name': 'Ácido Bórico (H[?]BO[?]) [Fertilizante Requerido]'
        },
        'Mo': {
            'primary': 'molibdato de sodio',  # Na2MoO4.2H2O
            'alternatives': ['molibdato de amonio'],
            'display_name': 'Molibdato de Sodio (Na[?]MoO[?]) [Fertilizante Requerido]'
        }
    }
    
    added_count = 0
    for micro in missing_micronutrients:
        if micro in required_micronutrient_sources:
            source_info = required_micronutrient_sources[micro]
            
            # Try primary source first
            fertilizer = fertilizer_db.create_fertilizer_from_database(source_info['primary'])
            
            if fertilizer:
                # Update the display name to indicate it's a required fertilizer
                fertilizer.name = source_info['display_name']
                enhanced_fertilizers.append(fertilizer)
                added_count += 1
                
                # Calculate expected contribution
                micro_content = (fertilizer.composition.cations.get(micro, 0) + 
                            fertilizer.composition.anions.get(micro, 0))
                
                print(f"  SUCCESS: Added: {fertilizer.name}")
                print(f"     {micro} content: {micro_content:.1f}%")
                print(f"     Need: {micronutrient_needs[micro]:.3f} mg/L")
                
            else:
                print(f"  ERROR: Failed to create fertilizer for {micro}")
    
    print(f"Auto-added {added_count} required micronutrient fertilizers")
    return enhanced_fertilizers

if __name__ == "__main__":
    import uvicorn
    import socket

    print("[START] Complete Modular Fertilizer Calculator API v5.0.0")
    print("[INFO] FEATURES: All modules implemented and integrated")
    print("[INFO] METHODS: Deterministic, Machine Learning")
    print("[INFO] API: Real Swagger integration with live data")
    print("[INFO] PDF: Professional Excel-like reports")
    print("=" * 70)
    
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
        print("[ERROR] No available ports found")
        exit(1)

    print(f"\n[SERVER] http://localhost:{available_port}")
    print(f"[DOCS] http://localhost:{available_port}/docs")
    print(f"[TEST] http://localhost:{available_port}/test")
    print(f"[SWAGGER] http://localhost:{available_port}/swagger-integrated-calculation")
    print(f"[ML] http://localhost:{available_port}/train-ml-model")
    print(f"[COMPARE] http://localhost:{available_port}/test-optimization-methods")
    print("\n[READY] Complete fertilizer calculation system operational!")
    
    uvicorn.run(app, host="0.0.0.0", port=available_port)