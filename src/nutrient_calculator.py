#!/usr/bin/env python3
"""
ENHANCED FERTILIZER CALCULATOR WITH MICRONUTRIENT SUPPORT
Complete solution including micronutrients in calculations, ML training, and PDF generation
"""

from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

class EnhancedFertilizerCalculator:
    """Enhanced calculator with complete micronutrient support"""
    
    def __init__(self):
        # Import the enhanced database
        from fertilizer_database import EnhancedFertilizerDatabase
        self.fertilizer_db = EnhancedFertilizerDatabase()
        
        # Import other components
        from verification_analyzer import SolutionVerifier, CostAnalyzer
        
        # Initialize nutrient calculator methods directly (no separate instance needed)
        self.nutrient_calc = self
        self.verifier = SolutionVerifier()
        self.cost_analyzer = CostAnalyzer()
        
        print("Enhanced Fertilizer Calculator initialized with micronutrient support")

    def get_complete_fertilizer_set(self, include_micronutrients: bool = True) -> List:
        """Get a complete set of fertilizers including micronutrients"""
        
        # Essential macronutrient fertilizers
        essential_fertilizers = [
            'nitrato de calcio',
            'nitrato de potasio', 
            'fosfato monopotasico',
            'sulfato de magnesio',
            'sulfato de potasio',
            'fosfato monoamonico',
            'cloruro de potasio'
        ]
        
        # Essential micronutrient fertilizers
        micronutrient_fertilizers = [
            'quelato de hierro',    # Fe-EDTA - best iron source
            'sulfato de manganeso', # MnSO4 - manganese source
            'sulfato de zinc',      # ZnSO4 - zinc source
            'sulfato de cobre',     # CuSO4 - copper source
            'acido borico',         # H3BO3 - boron source
            'molibdato de sodio'    # Na2MoO4 - molybdenum source
        ]
        
        fertilizer_list = []
        
        # Add macronutrient fertilizers
        for name in essential_fertilizers:
            fert = self.fertilizer_db.create_fertilizer_from_database(name)
            if fert:
                fertilizer_list.append(fert)
                print(f"  [SUCCESS] Added macronutrient: {fert.name}")
        
        # Add micronutrient fertilizers if requested
        if include_micronutrients:
            for name in micronutrient_fertilizers:
                fert = self.fertilizer_db.create_fertilizer_from_database(name)
                if fert:
                    fertilizer_list.append(fert)
                    # Show micronutrient content
                    micro_content = []
                    for elem in ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']:
                        content = fert.composition.cations.get(elem, 0) + fert.composition.anions.get(elem, 0)
                        if content > 0.1:
                            micro_content.append(f"{elem}:{content:.1f}%")
                    
                    print(f"  [TEST] Added micronutrient: {fert.name} ({', '.join(micro_content)})")
        
        print(f"\n[FORM] Complete fertilizer set: {len(fertilizer_list)} fertilizers")
        print(f"   Macronutrients: {len(essential_fertilizers)}")
        if include_micronutrients:
            print(f"   Micronutrients: {len(micronutrient_fertilizers)}")
        
        return fertilizer_list

    def create_complete_targets_with_micronutrients(self, base_targets: Dict[str, float]) -> Dict[str, float]:
        """Create complete target concentrations including micronutrients"""
        
        complete_targets = base_targets.copy()
        
        # Add standard micronutrient targets if not present
        standard_micronutrients = {
            'Fe': 2.0,    # Iron: 1.0-3.0 mg/L
            'Mn': 0.5,    # Manganese: 0.3-0.8 mg/L
            'Zn': 0.3,    # Zinc: 0.1-0.5 mg/L
            'Cu': 0.1,    # Copper: 0.05-0.2 mg/L
            'B': 0.5,     # Boron: 0.2-0.8 mg/L
            'Mo': 0.05    # Molybdenum: 0.01-0.1 mg/L
        }
        
        for micronutrient, default_value in standard_micronutrients.items():
            if micronutrient not in complete_targets:
                complete_targets[micronutrient] = default_value
                print(f"  [TARGET] Added micronutrient target: {micronutrient} = {default_value} mg/L")
        
        return complete_targets

    def optimize_with_micronutrients(self, targets: Dict[str, float], 
                                   water: Dict[str, float],
                                   fertilizers: List = None) -> Dict[str, float]:
        """Enhanced optimization that includes micronutrients"""
        
        print(f"\n=== ENHANCED OPTIMIZATION WITH MICRONUTRIENTS ===")
        
        # Get complete fertilizer set if not provided
        if fertilizers is None:
            fertilizers = self.get_complete_fertilizer_set(include_micronutrients=True)
        
        # Ensure complete targets
        complete_targets = self.create_complete_targets_with_micronutrients(targets)
        
        # Step 1: Calculate remaining nutrients after water
        remaining_nutrients = {}
        for element, target in complete_targets.items():
            water_content = water.get(element, 0)
            remaining = max(0, target - water_content)
            remaining_nutrients[element] = remaining
            
            if remaining > 0:
                print(f"  [TARGET] {element}: Target={target:.3f}, Water={water_content:.3f}, Need={remaining:.3f} mg/L")

        results = {}
        
        # Step 2: Macronutrients first (existing logic)
        print(f"\n[FORM] STEP 1: MACRONUTRIENTS")
        
        # Phosphorus sources
        if remaining_nutrients.get('P', 0) > 0:
            p_fertilizers = [f for f in fertilizers if f.composition.anions.get('P', 0) > 5]
            if p_fertilizers:
                # Prefer monopotassium phosphate
                mkp_ferts = [f for f in p_fertilizers if 'monopotasic' in f.name.lower()]
                best_p_fert = mkp_ferts[0] if mkp_ferts else p_fertilizers[0]
                
                p_needed = remaining_nutrients['P']
                dosage = self._calculate_dosage_for_element(best_p_fert, 'P', p_needed)
                
                if dosage > 0:
                    results[best_p_fert.name] = dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_p_fert, dosage)
                    print(f"    [SUCCESS] {best_p_fert.name}: {dosage/1000:.3f} g/L")
        
        # Calcium sources
        if remaining_nutrients.get('Ca', 0) > 0:
            ca_fertilizers = [f for f in fertilizers if f.composition.cations.get('Ca', 0) > 10]
            if ca_fertilizers:
                best_ca_fert = max(ca_fertilizers, key=lambda f: f.composition.cations.get('Ca', 0))
                ca_needed = remaining_nutrients['Ca']
                dosage = self._calculate_dosage_for_element(best_ca_fert, 'Ca', ca_needed)
                
                if dosage > 0:
                    results[best_ca_fert.name] = dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_ca_fert, dosage)
                    print(f"    [SUCCESS] {best_ca_fert.name}: {dosage/1000:.3f} g/L")
        
        # Potassium sources
        if remaining_nutrients.get('K', 0) > 0:
            k_fertilizers = [f for f in fertilizers if f.composition.cations.get('K', 0) > 20 and f.name not in results]
            if k_fertilizers:
                best_k_fert = max(k_fertilizers, key=lambda f: f.composition.cations.get('K', 0))
                k_needed = remaining_nutrients['K']
                dosage = self._calculate_dosage_for_element(best_k_fert, 'K', k_needed)
                
                if dosage > 0:
                    results[best_k_fert.name] = dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_k_fert, dosage)
                    print(f"    [SUCCESS] {best_k_fert.name}: {dosage/1000:.3f} g/L")
        
        # Magnesium sources
        if remaining_nutrients.get('Mg', 0) > 0:
            mg_fertilizers = [f for f in fertilizers if f.composition.cations.get('Mg', 0) > 5 and f.name not in results]
            if mg_fertilizers:
                best_mg_fert = max(mg_fertilizers, key=lambda f: f.composition.cations.get('Mg', 0))
                mg_needed = remaining_nutrients['Mg']
                dosage = self._calculate_dosage_for_element(best_mg_fert, 'Mg', mg_needed)
                
                if dosage > 0:
                    results[best_mg_fert.name] = dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_mg_fert, dosage)
                    print(f"    [SUCCESS] {best_mg_fert.name}: {dosage/1000:.3f} g/L")
        
        # Step 3: MICRONUTRIENTS (NEW!)
        print(f"\n[TEST] STEP 2: MICRONUTRIENTS")
        
        micronutrients = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        for micronutrient in micronutrients:
            need = remaining_nutrients.get(micronutrient, 0)
            if need > 0.01:  # Need at least 0.01 mg/L
                
                # Find best source for this micronutrient
                micro_fertilizers = [f for f in fertilizers 
                                   if f.composition.cations.get(micronutrient, 0) + 
                                      f.composition.anions.get(micronutrient, 0) > 0.5]
                
                if micro_fertilizers:
                    # Prefer chelates for Fe, Mn, Zn, Cu; direct sources for B, Mo
                    if micronutrient in ['Fe', 'Mn', 'Zn', 'Cu']:
                        chelate_sources = [f for f in micro_fertilizers if 'quelato' in f.name.lower()]
                        best_micro_fert = chelate_sources[0] if chelate_sources else micro_fertilizers[0]
                    else:
                        best_micro_fert = micro_fertilizers[0]
                    
                    # Calculate dosage
                    dosage = self._calculate_dosage_for_element(best_micro_fert, micronutrient, need)
                    
                    # Limit micronutrient dosages to reasonable amounts
                    max_micro_dosage = 50.0  # mg/L maximum
                    if dosage > max_micro_dosage:
                        dosage = max_micro_dosage
                    
                    if dosage > 0.1:  # Minimum meaningful dosage
                        results[best_micro_fert.name] = dosage / 1000.0
                        self._update_remaining_nutrients(remaining_nutrients, best_micro_fert, dosage)
                        print(f"    [TEST] {best_micro_fert.name}: {dosage/1000:.4f} g/L for {micronutrient}")
                
                else:
                    print(f"    [WARNING]  No source found for {micronutrient}")
        
        # Step 4: Final verification
        total_dosage = sum(results.values())
        active_fertilizers = len([d for d in results.values() if d > 0.001])
        
        print(f"\n[SUCCESS] ENHANCED OPTIMIZATION COMPLETE")
        print(f"   Active fertilizers: {active_fertilizers}")
        print(f"   Total dosage: {total_dosage:.3f} g/L")
        print(f"   Macronutrients: {len([f for f in results.keys() if not any(micro in f.lower() for micro in ['hierro', 'iron', 'manganeso', 'zinc', 'cobre', 'copper', 'borico', 'molibdato'])])}")
        print(f"   Micronutrients: {len([f for f in results.keys() if any(micro in f.lower() for micro in ['hierro', 'iron', 'manganeso', 'zinc', 'cobre', 'copper', 'borico', 'molibdato'])])}")
        
        return results

    
    def analyze_micronutrient_coverage(self, 
                                 fertilizers: List,
                                 target_concentrations: Dict[str, float],
                                 water_analysis: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze micronutrient coverage and identify gaps
        """
        print(f"[MICRO] Analyzing micronutrient coverage...")
        
        micronutrients = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        coverage_analysis = {
            'micronutrients_needed': {},
            'micronutrients_covered': {},
            'total_coverage_score': 0.0,
            'missing_micronutrients': [],
            'adequate_micronutrients': []
        }
        
        for micro in micronutrients:
            target = target_concentrations.get(micro, 0)
            water_content = water_analysis.get(micro, 0)
            remaining_need = max(0, target - water_content)
            
            if remaining_need > 0.001:  # Need significant amount
                # Check available sources
                available_sources = []
                total_potential = 0
                
                for fert in fertilizers:
                    cation_content = fert.composition.cations.get(micro, 0)
                    anion_content = fert.composition.anions.get(micro, 0)
                    total_content = cation_content + anion_content
                    
                    if total_content > 0.1:  # Meaningful content
                        available_sources.append({
                            'fertilizer': fert.name,
                            'content_percent': total_content,
                            'max_contribution': 2.0 * total_content * fert.chemistry.purity / 100.0 * 1000.0 / 100.0
                        })
                        total_potential += available_sources[-1]['max_contribution']
                
                if total_potential >= remaining_need * 0.8:  # Can cover at least 80%
                    coverage_analysis['micronutrients_covered'][micro] = {
                        'remaining_need': remaining_need,
                        'potential_supply': total_potential,
                        'coverage_ratio': total_potential / remaining_need,
                        'sources': available_sources
                    }
                    coverage_analysis['adequate_micronutrients'].append(micro)
                else:
                    coverage_analysis['micronutrients_needed'][micro] = {
                        'remaining_need': remaining_need,
                        'potential_supply': total_potential,
                        'gap': remaining_need - total_potential,
                        'sources': available_sources
                    }
                    coverage_analysis['missing_micronutrients'].append(micro)
        
        # Calculate overall coverage score
        total_micros = len(micronutrients)
        covered_micros = len(coverage_analysis['adequate_micronutrients'])
        coverage_analysis['total_coverage_score'] = (covered_micros / total_micros) * 100
        
        print(f"[MICRO] Coverage analysis complete:")
        print(f"  Adequate: {len(coverage_analysis['adequate_micronutrients'])}/{total_micros}")
        print(f"  Missing: {len(coverage_analysis['missing_micronutrients'])}/{total_micros}")
        print(f"  Score: {coverage_analysis['total_coverage_score']:.1f}%")
        
        return coverage_analysis
    
    def _update_remaining_nutrients(self, remaining_nutrients: Dict[str, float], 
                                  fertilizer, dosage_mg_l: float):
        """Update remaining nutrients after adding a fertilizer"""
        
        all_elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'S', 'Cl', 'P', 'HCO3', 
                       'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        for element in all_elements:
            cation_content = fertilizer.composition.cations.get(element, 0)
            anion_content = fertilizer.composition.anions.get(element, 0)
            total_content = cation_content + anion_content
            
            if total_content > 0:
                contribution = self.nutrient_calc.calculate_element_contribution(
                    dosage_mg_l, total_content, fertilizer.percentage
                )
                if element in remaining_nutrients:
                    remaining_nutrients[element] = max(0, remaining_nutrients[element] - contribution)

    def _calculate_enhanced_contributions(self, fertilizers: List, dosages: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate contributions including all micronutrients"""
        
        # Extended element list including micronutrients
        elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'S', 'Cl', 'P', 'HCO3', 
                   'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
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
                contributions[category][element] = round(contributions[category][element], 4)

        return contributions

    def _calculate_enhanced_water_contributions(self, water_analysis: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate water contributions including micronutrients"""
        
        elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'S', 'Cl', 'P', 'HCO3', 
                   'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        water_contrib = {
            'IONES_mg_L_DEL_AGUA': {},
            'mmol_L': {},
            'meq_L': {}
        }

        for element in elements:
            mg_l = water_analysis.get(element, 0)
            mmol_l = self.nutrient_calc.convert_mg_to_mmol(mg_l, element)
            meq_l = self.nutrient_calc.convert_mmol_to_meq(mmol_l, element)

            water_contrib['IONES_mg_L_DEL_AGUA'][element] = round(mg_l, 4)
            water_contrib['mmol_L'][element] = round(mmol_l, 4)
            water_contrib['meq_L'][element] = round(meq_l, 4)

        return water_contrib

    def _calculate_enhanced_final_solution(self, nutrient_contrib: Dict, water_contrib: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate final solution including micronutrients"""
        
        elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'N', 'S', 'Cl', 'P', 'HCO3', 
                   'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
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

            final['FINAL_mg_L'][element] = round(final_mg_l, 4)
            final['FINAL_mmol_L'][element] = round(final_mmol_l, 4)
            final['FINAL_meq_L'][element] = round(final_meq_l, 4)

        # Calculate EC and pH with enhanced algorithm
        cations = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'Fe', 'Mn', 'Zn', 'Cu']
        cation_sum = sum(final['FINAL_meq_L'].get(cation, 0) for cation in cations)
        ec = self.nutrient_calc.calculate_ec_advanced(final['FINAL_meq_L'])
        ph = self.nutrient_calc.calculate_ph_advanced(final['FINAL_mg_L'], final['FINAL_meq_L'])

        final['calculated_EC'] = round(ec, 2)
        final['calculated_pH'] = round(ph, 1)

        return final

    def _generate_micronutrient_summary(self, final_concentrations: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary of micronutrient concentrations"""
        
        micronutrients = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        optimal_ranges = {
            'Fe': {'min': 1.0, 'max': 3.0, 'optimal': 2.0},
            'Mn': {'min': 0.3, 'max': 0.8, 'optimal': 0.5},
            'Zn': {'min': 0.1, 'max': 0.5, 'optimal': 0.3},
            'Cu': {'min': 0.05, 'max': 0.2, 'optimal': 0.1},
            'B': {'min': 0.2, 'max': 0.8, 'optimal': 0.5},
            'Mo': {'min': 0.01, 'max': 0.1, 'optimal': 0.05}
        }
        
        summary = {
            'micronutrient_status': {},
            'total_micronutrients_supplied': 0,
            'deficient_micronutrients': [],
            'adequate_micronutrients': [],
            'excessive_micronutrients': []
        }
        
        for micro in micronutrients:
            concentration = final_concentrations.get(micro, 0)
            ranges = optimal_ranges[micro]
            
            # Determine status
            if concentration < ranges['min']:
                status = 'Deficient'
                summary['deficient_micronutrients'].append(micro)
            elif concentration > ranges['max']:
                status = 'Excessive'
                summary['excessive_micronutrients'].append(micro)
            else:
                status = 'Adequate'
                summary['adequate_micronutrients'].append(micro)
            
            summary['micronutrient_status'][micro] = {
                'concentration': round(concentration, 4),
                'target_range': f"{ranges['min']}-{ranges['max']}",
                'optimal': ranges['optimal'],
                'status': status,
                'adequacy_percent': round(min(concentration / ranges['optimal'], 2.0) * 100, 1)
            }
            
            if concentration > 0.001:
                summary['total_micronutrients_supplied'] += 1
        
        return summary

    # Missing calculation methods needed by ML optimizer and API
    def calculate_fertilizer_requirement(self, target_element: str, target_concentration: float,
                                         fertilizer_composition: Dict[str, float],
                                         purity: float, molecular_weight: float) -> float:
        """Calculate fertilizer requirement"""
        if target_element not in fertilizer_composition:
            return 0.0

        element_weight_percent = fertilizer_composition[target_element]
        if element_weight_percent <= 0:
            return 0.0

        # Calculate fertilizer amount in mg/L
        fertilizer_amount = target_concentration * 100.0 / (element_weight_percent * (purity / 100.0))
        return max(0, fertilizer_amount)

    def calculate_element_contribution(self, fertilizer_amount: float, element_weight_percent: float,
                                       purity: float) -> float:
        """Calculate element contribution from fertilizer amount"""
        if fertilizer_amount <= 0 or element_weight_percent <= 0:
            return 0.0

        contribution = fertilizer_amount * element_weight_percent * (purity / 100.0) / 100.0
        return contribution

    def convert_mg_to_mmol(self, mg_l: float, element: str) -> float:
        """Convert mg/L to mmol/L"""
        element_data = {
            'N': 14.01, 'P': 30.97, 'K': 39.10, 'Ca': 40.08, 'Mg': 24.31, 'S': 32.06,
            'Cl': 35.45, 'Na': 22.99, 'NH4': 18.04, 'HCO3': 61.02,
            'Fe': 55.85, 'Mn': 54.94, 'Zn': 65.38, 'Cu': 63.55, 'B': 10.81, 'Mo': 95.96
        }
        
        if element in element_data and mg_l > 0:
            return mg_l / element_data[element]
        return 0.0

    def convert_mmol_to_meq(self, mmol_l: float, element: str) -> float:
        """Convert mmol/L to meq/L"""
        element_valences = {
            'N': 1, 'P': 3, 'K': 1, 'Ca': 2, 'Mg': 2, 'S': 2, 'Cl': 1, 'Na': 1, 'NH4': 1, 'HCO3': 1,
            'Fe': 2, 'Mn': 2, 'Zn': 2, 'Cu': 2, 'B': 3, 'Mo': 6
        }
        
        if element in element_valences and mmol_l > 0:
            return mmol_l * element_valences[element]
        return 0.0

    def convert_mg_to_meq_direct(self, mg_l: float, element: str) -> float:
        """
        Direct conversion from mg/L to meq/L for ionic balance calculations
        Used by linear programming optimizer for precise ionic balance
        """
        
        # Molecular weights and valences for conversion
        element_properties = {
            'Ca': {'mw': 40.08, 'valence': 2},
            'K': {'mw': 39.10, 'valence': 1}, 
            'Mg': {'mw': 24.31, 'valence': 2},
            'Na': {'mw': 22.99, 'valence': 1},
            'NH4': {'mw': 18.04, 'valence': 1},
            'Fe': {'mw': 55.85, 'valence': 2},
            'Mn': {'mw': 54.94, 'valence': 2},
            'Zn': {'mw': 65.38, 'valence': 2},
            'Cu': {'mw': 63.55, 'valence': 2},
            'NO3': {'mw': 62.00, 'valence': 1},
            'H2PO4': {'mw': 96.99, 'valence': 1},
            'SO4': {'mw': 96.06, 'valence': 2},
            'Cl': {'mw': 35.45, 'valence': 1},
            'HCO3': {'mw': 61.02, 'valence': 1}
        }
        
        if element not in element_properties:
            return 0.0
        
        props = element_properties[element]
        mmol_l = mg_l / props['mw']
        meq_l = mmol_l * props['valence']
        
        return meq_l


    def calculate_ec_advanced(self, final_meq_l: Dict[str, float]) -> float:
        """Calculate electrical conductivity from meq/L values"""
        total_meq = sum(final_meq_l.values())
        return total_meq * 0.1  # Simple approximation

    def calculate_ph_advanced(self, final_mg_l: Dict[str, float], final_meq_l: Dict[str, float]) -> float:
        """Calculate pH from solution composition"""
        return 6.0  # Simple default - would need more complex calculation

    def get_enhanced_fertilizer_recommendations(self) -> Dict[str, List[str]]:
        """Get recommendations for complete fertilizer programs"""
        
        return {
            'essential_macronutrients': [
                'Nitrato de Calcio - Primary calcium and nitrogen source',
                'Nitrato de Potasio - Primary potassium and nitrogen source',
                'Fosfato Monopotásico - Primary phosphorus source',
                'Sulfato de Magnesio - Primary magnesium and sulfur source'
            ],
            'essential_micronutrients': [
                'Quelato de Hierro (Fe-EDTA) - Best iron source for hydroponics',
                'Sulfato de Manganeso - Reliable manganese source',
                'Sulfato de Zinc - Primary zinc source',
                'Sulfato de Cobre - Copper supplementation',
                'Ácido Bórico - Boron source',
                'Molibdato de Sodio - Molybdenum source'
            ],
            'alternative_sources': [
                'Iron: Sulfato de Hierro (less stable but cheaper)',
                'Manganese: Cloruro de Manganeso (alternative form)',
                'Zinc: Quelato de Zinc (premium chelated form)',
                'Copper: Quelato de Cobre (premium chelated form)',
                'Boron: Borax (alternative boron source)'
            ],
            'usage_tips': [
                'Use chelated forms (EDTA) for Fe, Mn, Zn, Cu in alkaline water',
                'Prepare micronutrient stock solutions separately',
                'Monitor micronutrient levels weekly',
                'Adjust dosages based on plant growth stage',
                'Store micronutrient fertilizers in cool, dark conditions'
            ]
        }

    def analyze_micronutrient_coverage(self, fertilizers: List, targets: Dict[str, float], 
                                      water: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze micronutrient coverage and identify gaps that need supplementation
        """
        print(f"\nANALYZING MICRONUTRIENT COVERAGE")
        
        micronutrients = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        coverage_analysis = {
            'micronutrients_needed': {},
            'available_sources': {},
            'missing_micronutrients': [],
            'coverage_percentage': 0,
            'recommendations': []
        }
        
        # Calculate micronutrient needs
        for micro in micronutrients:
            target = targets.get(micro, 0)
            water_content = water.get(micro, 0)
            remaining_need = max(0, target - water_content)
            
            if remaining_need > 0.01:
                coverage_analysis['micronutrients_needed'][micro] = {
                    'target': target,
                    'water_content': water_content,
                    'remaining_need': remaining_need,
                    'available_sources': []
                }
        
        # Check available sources in fertilizer list
        for fertilizer in fertilizers:
            for micro in micronutrients:
                if micro in coverage_analysis['micronutrients_needed']:
                    cation_content = fertilizer.composition.cations.get(micro, 0)
                    anion_content = fertilizer.composition.anions.get(micro, 0)
                    total_content = cation_content + anion_content
                    
                    if total_content > 0.1:  # Significant content
                        coverage_analysis['micronutrients_needed'][micro]['available_sources'].append({
                            'fertilizer_name': fertilizer.name,
                            'content_percent': total_content,
                            'is_required_supplement': '[Fertilizante Requerido]' in fertilizer.name
                        })
        
        # Identify missing micronutrients
        for micro, need_info in coverage_analysis['micronutrients_needed'].items():
            if not need_info['available_sources']:
                coverage_analysis['missing_micronutrients'].append(micro)
        
        # Calculate coverage percentage
        total_needed = len(coverage_analysis['micronutrients_needed'])
        covered = total_needed - len(coverage_analysis['missing_micronutrients'])
        coverage_analysis['coverage_percentage'] = (covered / total_needed * 100) if total_needed > 0 else 100
        
        # Generate recommendations
        if coverage_analysis['missing_micronutrients']:
            coverage_analysis['recommendations'].append(
                f"Add fertilizers for missing micronutrients: {', '.join(coverage_analysis['missing_micronutrients'])}"
            )
        
        if coverage_analysis['coverage_percentage'] < 100:
            coverage_analysis['recommendations'].append(
                "Consider using micronutrient fertilizer mix for complete coverage"
            )
        
        print(f"   Micronutrients needed: {len(coverage_analysis['micronutrients_needed'])}")
        print(f"   Coverage: {coverage_analysis['coverage_percentage']:.1f}%")
        print(f"   Missing: {coverage_analysis['missing_micronutrients']}")
        
        return coverage_analysis

    
    def calculate_micronutrient_dosages(self, 
                                    micronutrient_needs: Dict[str, float],
                                    fertilizers: List) -> Dict[str, float]:
        """
        Calculate specific dosages for micronutrient requirements
        """
        print(f"[MICRO] Calculating micronutrient dosages...")
        
        micronutrient_dosages = {}
        
        # Priority order for micronutrient fulfillment
        priority_order = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        for micro in priority_order:
            need = micronutrient_needs.get(micro, 0)
            if need <= 0.001:
                continue
            
            print(f"[MICRO] Processing {micro}: need {need:.3f} mg/L")
            
            # Find best fertilizer for this micronutrient
            best_fertilizer = None
            best_efficiency = 0
            
            for fert in fertilizers:
                cation_content = fert.composition.cations.get(micro, 0)
                anion_content = fert.composition.anions.get(micro, 0)
                total_content = cation_content + anion_content
                
                if total_content > 0.1:  # Has meaningful micronutrient content
                    # Calculate efficiency (content per gram)
                    efficiency = total_content
                    
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_fertilizer = fert
            
            if best_fertilizer:
                # Calculate required dosage
                cation_content = best_fertilizer.composition.cations.get(micro, 0)
                anion_content = best_fertilizer.composition.anions.get(micro, 0)
                total_content = cation_content + anion_content
                
                # Calculate dosage needed (mg/L fertilizer)
                required_dosage_mg_l = (need * 100 * 100) / (total_content * best_fertilizer.chemistry.purity)
                required_dosage_g_l = required_dosage_mg_l / 1000
                
                # Apply reasonable limits
                max_dosage = min(required_dosage_g_l, 1.5)  # Max 1.5 g/L for micronutrients
                
                micronutrient_dosages[best_fertilizer.name] = max_dosage
                
                print(f"  Solution: {max_dosage:.3f} g/L of {best_fertilizer.name}")
            else:
                print(f"  Warning: No suitable fertilizer found for {micro}")
        
        return micronutrient_dosages

    def validate_micronutrient_solution(self,
                                    final_concentrations: Dict[str, float],
                                    target_concentrations: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate micronutrient solution against targets and optimal ranges
        """
        print(f"[MICRO] Validating micronutrient solution...")
        
        micronutrients = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        # Define optimal ranges for micronutrients (mg/L)
        optimal_ranges = {
            'Fe': {'min': 1.0, 'max': 5.0, 'optimal': 2.0},
            'Mn': {'min': 0.1, 'max': 2.0, 'optimal': 0.5},
            'Zn': {'min': 0.05, 'max': 1.0, 'optimal': 0.3},
            'Cu': {'min': 0.02, 'max': 0.2, 'optimal': 0.05},
            'B': {'min': 0.1, 'max': 1.0, 'optimal': 0.3},
            'Mo': {'min': 0.01, 'max': 0.1, 'optimal': 0.05}
        }
        
        validation_results = {
            'micronutrient_status': {},
            'total_micronutrients_supplied': 0,
            'deficient_micronutrients': [],
            'adequate_micronutrients': [],
            'excessive_micronutrients': []
        }
        
        for micro in micronutrients:
            concentration = final_concentrations.get(micro, 0)
            target = target_concentrations.get(micro, 0)
            ranges = optimal_ranges[micro]
            
            # Determine status
            if concentration < ranges['min']:
                status = 'Deficient'
                validation_results['deficient_micronutrients'].append(micro)
            elif concentration > ranges['max']:
                status = 'Excessive'
                validation_results['excessive_micronutrients'].append(micro)
            else:
                status = 'Adequate'
                validation_results['adequate_micronutrients'].append(micro)
            
            # Calculate deviation from target
            if target > 0:
                deviation_percent = ((concentration - target) / target) * 100
            else:
                deviation_percent = 0
            
            validation_results['micronutrient_status'][micro] = {
                'concentration': round(concentration, 4),
                'target': round(target, 4),
                'deviation_percent': round(deviation_percent, 2),
                'target_range': f"{ranges['min']}-{ranges['max']}",
                'optimal': ranges['optimal'],
                'status': status,
                'adequacy_percent': round(min(concentration / ranges['optimal'], 2.0) * 100, 1)
            }
            
            if concentration > 0.001:
                validation_results['total_micronutrients_supplied'] += 1
        
        print(f"[MICRO] Validation complete:")
        print(f"  Adequate: {len(validation_results['adequate_micronutrients'])}")
        print(f"  Deficient: {len(validation_results['deficient_micronutrients'])}")
        print(f"  Excessive: {len(validation_results['excessive_micronutrients'])}")
        
        return validation_results

    def validate_micronutrient_solution(self, final_concentrations: Dict[str, float], 
                                       targets: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate final micronutrient concentrations against targets and safety limits
        """
        print(f"\n[SUCCESS] VALIDATING MICRONUTRIENT SOLUTION")
        
        micronutrients = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        validation_results = {
            'micronutrient_status': {},
            'safety_warnings': [],
            'adequacy_warnings': [],
            'overall_status': 'adequate',
            'compliance_score': 0
        }
        
        # Safety limits (mg/L) - levels above which toxicity may occur
        safety_limits = {
            'Fe': 5.0,   # Iron toxicity above 5 mg/L
            'Mn': 2.0,   # Manganese toxicity above 2 mg/L
            'Zn': 1.0,   # Zinc toxicity above 1 mg/L
            'Cu': 0.5,   # Copper very toxic above 0.5 mg/L
            'B': 1.5,    # Boron narrow range, toxic above 1.5 mg/L
            'Mo': 0.2    # Molybdenum toxic above 0.2 mg/L
        }
        
        # Adequacy ranges (mg/L) - minimum levels for plant health
        adequacy_minimums = {
            'Fe': 1.0,   # Iron deficiency below 1.0 mg/L
            'Mn': 0.2,   # Manganese deficiency below 0.2 mg/L
            'Zn': 0.05,  # Zinc deficiency below 0.05 mg/L
            'Cu': 0.02,  # Copper deficiency below 0.02 mg/L
            'B': 0.1,    # Boron deficiency below 0.1 mg/L
            'Mo': 0.005  # Molybdenum deficiency below 0.005 mg/L
        }
        
        compliant_micronutrients = 0
        
        for micro in micronutrients:
            target = targets.get(micro, 0)
            final = final_concentrations.get(micro, 0)
            safety_limit = safety_limits[micro]
            adequacy_min = adequacy_minimums[micro]
            
            # Determine status
            if final > safety_limit:
                status = 'toxic'
                validation_results['safety_warnings'].append(
                    f"{micro}: {final:.3f} mg/L exceeds safety limit ({safety_limit} mg/L)"
                )
                validation_results['overall_status'] = 'unsafe'
            elif final < adequacy_min:
                status = 'deficient'
                validation_results['adequacy_warnings'].append(
                    f"{micro}: {final:.3f} mg/L below adequacy minimum ({adequacy_min} mg/L)"
                )
            elif target > 0 and abs(final - target) / target <= 0.20:  # Within 20% of target
                status = 'adequate'
                compliant_micronutrients += 1
            elif target > 0 and abs(final - target) / target <= 0.50:  # Within 50% of target
                status = 'acceptable'
                compliant_micronutrients += 0.5
            else:
                status = 'off_target'
            
            validation_results['micronutrient_status'][micro] = {
                'target': target,
                'final': final,
                'status': status,
                'safety_limit': safety_limit,
                'adequacy_minimum': adequacy_min,
                'deviation_percent': ((final - target) / target * 100) if target > 0 else 0
            }
            
            print(f"   {micro}: {final:.3f} mg/L (target: {target:.3f}) - {status.upper()}")
        
        # Calculate compliance score
        total_micronutrients = len([m for m in micronutrients if targets.get(m, 0) > 0])
        validation_results['compliance_score'] = (compliant_micronutrients / total_micronutrients * 100) if total_micronutrients > 0 else 100
        
        print(f"   Overall status: {validation_results['overall_status'].upper()}")
        print(f"   Compliance score: {validation_results['compliance_score']:.1f}%")
        
        if validation_results['safety_warnings']:
            print(f"   [WARNING]  Safety warnings: {len(validation_results['safety_warnings'])}")
        
        if validation_results['adequacy_warnings']:
            print(f"   [WARNING]  Adequacy warnings: {len(validation_results['adequacy_warnings'])}")
        
        return validation_results

    def generate_micronutrient_recommendations(self, validation_results: Dict[str, Any], 
                                             fertilizers: List) -> List[str]:
        """
        Generate specific recommendations for micronutrient management
        """
        recommendations = []
        
        # Safety recommendations
        for warning in validation_results['safety_warnings']:
            micro = warning.split(':')[0]
            recommendations.append(f"[?] URGENT: Reduce {micro} fertilizer dosage to prevent toxicity")
        
        # Adequacy recommendations  
        for warning in validation_results['adequacy_warnings']:
            micro = warning.split(':')[0]
            
            # Find if we have a fertilizer for this micronutrient
            has_fertilizer = any(
                f.composition.cations.get(micro, 0) + f.composition.anions.get(micro, 0) > 0.1
                for f in fertilizers
            )
            
            if has_fertilizer:
                recommendations.append(f"[UP] Increase {micro} fertilizer dosage to meet adequacy requirements")
            else:
                recommendations.append(f"[PLUS] Add {micro} fertilizer source (auto-supplementation recommended)")
        
        # Performance recommendations
        compliance_score = validation_results['compliance_score']
        if compliance_score >= 90:
            recommendations.append("[SUCCESS] Excellent micronutrient balance achieved")
        elif compliance_score >= 70:
            recommendations.append("[CHART] Good micronutrient coverage with minor adjustments needed")
        elif compliance_score >= 50:
            recommendations.append("[WARNING] Moderate micronutrient gaps - consider formula review")
        else:
            recommendations.append("[RED] Significant micronutrient deficiencies - major formula revision needed")
        
        # Water quality considerations
        recommendations.append("[WATER] Monitor water quality for micronutrient interactions")
        recommendations.append("[MICRO] Consider chelated forms for hard water conditions")
        
        # Application recommendations
        recommendations.append("[PACKAGE] Prepare micronutrient stock solution separately from macronutrients")
        recommendations.append("[TEMP] Store micronutrient solutions in cool, dark conditions")
        recommendations.append("[TIME] Replace micronutrient stock solutions every 2-4 weeks")
        
        # pH management
        recommendations.append("[RULER] Maintain solution pH 5.8-6.2 for optimal micronutrient availability")
        
        return recommendations[:10]  # Limit to top 10 recommendations

    # Factory function
    def create_enhanced_calculator():
        """Create enhanced calculator with micronutrient support"""
        return EnhancedFertilizerCalculator()

    def calculate_fertilizer_contribution_matrix(self, fertilizers: List, nutrients: List[str]) -> np.ndarray:
        """
        Calculate contribution matrix for linear programming optimization
        Returns matrix A where A[i,j] = contribution of fertilizer j to nutrient i per g/L
        """
        
        n_nutrients = len(nutrients)
        n_fertilizers = len(fertilizers)
        
        # Initialize contribution matrix
        contribution_matrix = np.zeros((n_nutrients, n_fertilizers))
        
        for i, nutrient in enumerate(nutrients):
            for j, fertilizer in enumerate(fertilizers):
                # Get nutrient content from fertilizer composition
                cation_content = fertilizer.composition.cations.get(nutrient, 0.0)
                anion_content = fertilizer.composition.anions.get(nutrient, 0.0)
                total_content = cation_content + anion_content
                
                if total_content > 0:
                    # Calculate contribution factor: g/L → mg/L
                    # dosage_g_L * content_% * purity_% / 100 * 1000_mg/g / 100_%
                    contribution_factor = (total_content * fertilizer.chemistry.purity / 100.0 * 
                                        1000.0 / 100.0)
                    contribution_matrix[i, j] = contribution_factor
        
        return contribution_matrix

    def validate_linear_programming_solution(self, 
                                        dosages: Dict[str, float],
                                        targets: Dict[str, float],
                                        water: Dict[str, float],
                                        fertilizers: List) -> Dict[str, Any]:
        """
        Validate linear programming solution for feasibility and quality
        """
        
        print(f"[LP] Validating linear programming solution...")
        
        # Calculate achieved concentrations
        achieved = self.calculate_achieved_concentrations_from_dosages(
            dosages, water, fertilizers
        )
        
        # Calculate deviations
        deviations = {}
        total_absolute_deviation = 0
        nutrient_count = 0
        
        for nutrient, target in targets.items():
            if target > 0:
                achieved_val = achieved.get(nutrient, 0)
                deviation_percent = ((achieved_val - target) / target) * 100
                deviations[nutrient] = deviation_percent
                total_absolute_deviation += abs(deviation_percent)
                nutrient_count += 1
        
        average_absolute_deviation = total_absolute_deviation / max(nutrient_count, 1)
        
        # Check ionic balance
        ionic_balance_error = self.calculate_ionic_balance_error_detailed(achieved)
        
        # Check dosage constraints
        active_fertilizers = len([d for d in dosages.values() if d > 0.001])
        total_dosage = sum(dosages.values())
        max_individual_dosage = max(dosages.values()) if dosages else 0
        
        # Feasibility checks
        feasibility_issues = []
        
        if total_dosage > 15.0:
            feasibility_issues.append(f"Total dosage too high: {total_dosage:.2f} g/L > 15.0 g/L")
        
        if max_individual_dosage > 5.0:
            feasibility_issues.append(f"Individual dosage too high: {max_individual_dosage:.2f} g/L > 5.0 g/L")
        
        if ionic_balance_error > 20.0:
            feasibility_issues.append(f"Ionic balance error too high: {ionic_balance_error:.1f}% > 20%")
        
        if average_absolute_deviation > 25.0:
            feasibility_issues.append(f"Average deviation too high: {average_absolute_deviation:.1f}% > 25%")
        
        # Quality assessment
        quality_score = 100.0
        
        # Penalize deviations
        quality_score -= min(average_absolute_deviation, 50.0)
        
        # Penalize ionic imbalance
        quality_score -= min(ionic_balance_error, 30.0)
        
        # Penalize excessive dosages
        if total_dosage > 10.0:
            quality_score -= (total_dosage - 10.0) * 5
        
        quality_score = max(0.0, quality_score)
        
        # Performance classification
        if quality_score >= 90 and average_absolute_deviation <= 2.0:
            performance_class = "Excellent"
        elif quality_score >= 80 and average_absolute_deviation <= 5.0:
            performance_class = "Good"
        elif quality_score >= 60 and average_absolute_deviation <= 15.0:
            performance_class = "Acceptable"
        else:
            performance_class = "Poor"
        
        return {
            'validation_status': 'Passed' if not feasibility_issues else 'Failed',
            'feasibility_issues': feasibility_issues,
            'performance_metrics': {
                'quality_score': quality_score,
                'performance_class': performance_class,
                'average_absolute_deviation': average_absolute_deviation,
                'ionic_balance_error': ionic_balance_error,
                'active_fertilizers': active_fertilizers,
                'total_dosage': total_dosage,
                'max_individual_dosage': max_individual_dosage
            },
            'achieved_concentrations': achieved,
            'deviations_percent': deviations,
            'nutrient_analysis': {
                'excellent_nutrients': len([d for d in deviations.values() if abs(d) <= 0.1]),
                'good_nutrients': len([d for d in deviations.values() if 0.1 < abs(d) <= 5.0]),
                'acceptable_nutrients': len([d for d in deviations.values() if 5.0 < abs(d) <= 15.0]),
                'poor_nutrients': len([d for d in deviations.values() if abs(d) > 15.0])
            }
        }

    def calculate_achieved_concentrations_from_dosages(self,
                                                    dosages: Dict[str, float],
                                                    water: Dict[str, float],
                                                    fertilizers: List) -> Dict[str, float]:
        """
        Calculate achieved nutrient concentrations from fertilizer dosages
        Optimized for linear programming validation
        """
        
        achieved = water.copy()
        fert_map = {f.name: f for f in fertilizers}
        
        for fert_name, dosage_g_l in dosages.items():
            if dosage_g_l > 0 and fert_name in fert_map:
                fertilizer = fert_map[fert_name]
                dosage_mg_l = dosage_g_l * 1000
                
                # Add contributions from cations
                for element, content_percent in fertilizer.composition.cations.items():
                    if content_percent > 0:
                        contribution = self.calculate_element_contribution(
                            dosage_mg_l, content_percent, fertilizer.chemistry.purity
                        )
                        achieved[element] = achieved.get(element, 0) + contribution
                
                # Add contributions from anions
                for element, content_percent in fertilizer.composition.anions.items():
                    if content_percent > 0:
                        contribution = self.calculate_element_contribution(
                            dosage_mg_l, content_percent, fertilizer.chemistry.purity
                        )
                        achieved[element] = achieved.get(element, 0) + contribution
        
        return achieved

    def calculate_ionic_balance_error_detailed(self, concentrations: Dict[str, float]) -> float:
        """
        Calculate detailed ionic balance error for linear programming validation
        """
        
        # Define ionic species
        cation_elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'Fe', 'Mn', 'Zn', 'Cu']
        anion_elements = ['NO3', 'H2PO4', 'SO4', 'Cl', 'HCO3']
        
        # Handle N → NO3 conversion
        working_concentrations = concentrations.copy()
        if 'N' in working_concentrations and 'NO3' not in working_concentrations:
            # Convert N to NO3 (N is typically in nitrate form)
            n_content = working_concentrations['N']
            # NO3 MW = 62, N MW = 14, so NO3 = N * (62/14)
            working_concentrations['NO3'] = n_content * (62.0 / 14.0)
        
        # Handle P → H2PO4 conversion
        if 'P' in working_concentrations and 'H2PO4' not in working_concentrations:
            # Convert P to H2PO4 (P is typically in phosphate form)
            p_content = working_concentrations['P']
            # H2PO4 MW = 97, P MW = 31, so H2PO4 = P * (97/31)
            working_concentrations['H2PO4'] = p_content * (97.0 / 31.0)
        
        # Handle S → SO4 conversion
        if 'S' in working_concentrations and 'SO4' not in working_concentrations:
            # Convert S to SO4 (S is typically in sulfate form)
            s_content = working_concentrations['S']
            # SO4 MW = 96, S MW = 32, so SO4 = S * (96/32)
            working_concentrations['SO4'] = s_content * (96.0 / 32.0)
        
        try:
            # Calculate cation sum in meq/L
            cation_meq = 0
            for element in cation_elements:
                concentration = working_concentrations.get(element, 0)
                if concentration > 0:
                    meq_contribution = self.convert_mg_to_meq_direct(concentration, element)
                    cation_meq += meq_contribution
            
            # Calculate anion sum in meq/L  
            anion_meq = 0
            for element in anion_elements:
                concentration = working_concentrations.get(element, 0)
                if concentration > 0:
                    meq_contribution = self.convert_mg_to_meq_direct(concentration, element)
                    anion_meq += meq_contribution
            
            # Calculate balance error
            if cation_meq > 0 or anion_meq > 0:
                balance_error = abs(cation_meq - anion_meq) / max(cation_meq, anion_meq, 1.0) * 100
            else:
                balance_error = 0.0
            
            return balance_error
            
        except Exception as e:
            print(f"[LP] Error calculating ionic balance: {e}")
            return 100.0  # High error if calculation fails

    def optimize_micronutrients_with_linear_programming(self,
                                                    base_dosages: Dict[str, float],
                                                    micronutrient_targets: Dict[str, float],
                                                    water: Dict[str, float],
                                                    all_fertilizers: List) -> Dict[str, float]:
        """
        Optimize micronutrient dosages using linear programming principles
        Focuses specifically on achieving micronutrient targets
        """
        
        print(f"[LP] Optimizing micronutrients with linear programming...")
        
        micronutrients = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        # Filter fertilizers that contribute to micronutrients
        micro_fertilizers = []
        for fert in all_fertilizers:
            has_micro = False
            for micro in micronutrients:
                cation_content = fert.composition.cations.get(micro, 0)
                anion_content = fert.composition.anions.get(micro, 0)
                if (cation_content + anion_content) > 0.1:  # Has meaningful micronutrient content
                    has_micro = True
                    break
            
            if has_micro:
                micro_fertilizers.append(fert)
        
        if not micro_fertilizers:
            print(f"[LP] No micronutrient fertilizers available")
            return base_dosages
        
        print(f"[LP] Found {len(micro_fertilizers)} micronutrient fertilizers")
        
        # Calculate current micronutrient concentrations from base dosages
        current_micros = {}
        for micro in micronutrients:
            current_micros[micro] = water.get(micro, 0)
            
            # Add contributions from base dosages
            for fert_name, dosage_g_l in base_dosages.items():
                if dosage_g_l > 0:
                    # Find fertilizer
                    fert = next((f for f in all_fertilizers if f.name == fert_name), None)
                    if fert:
                        dosage_mg_l = dosage_g_l * 1000
                        cation_content = fert.composition.cations.get(micro, 0)
                        anion_content = fert.composition.anions.get(micro, 0)
                        total_content = cation_content + anion_content
                        
                        if total_content > 0:
                            contribution = self.calculate_element_contribution(
                                dosage_mg_l, total_content, fert.chemistry.purity
                            )
                            current_micros[micro] += contribution
        
        # Calculate micronutrient needs
        micro_needs = {}
        for micro in micronutrients:
            target = micronutrient_targets.get(micro, 0)
            current = current_micros.get(micro, 0)
            need = max(0, target - current)
            micro_needs[micro] = need
            
            if need > 0:
                print(f"  {micro}: Need {need:.3f} mg/L (target: {target:.3f}, current: {current:.3f})")
        
        # If no micronutrient needs, return base dosages
        total_need = sum(micro_needs.values())
        if total_need <= 0.001:
            print(f"[LP] All micronutrients satisfied with current dosages")
            return base_dosages
        
        # Simple linear programming approach for micronutrients
        enhanced_dosages = base_dosages.copy()
        
        # Priority order for micronutrient sources
        micro_priority = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        for micro in micro_priority:
            need = micro_needs.get(micro, 0)
            if need <= 0.001:
                continue
            
            # Find best fertilizer for this micronutrient
            best_fert = None
            best_efficiency = 0
            
            for fert in micro_fertilizers:
                cation_content = fert.composition.cations.get(micro, 0)
                anion_content = fert.composition.anions.get(micro, 0)
                total_content = cation_content + anion_content
                
                if total_content > 0:
                    # Calculate efficiency (content per fertilizer cost)
                    efficiency = total_content
                    
                    # Prefer fertilizers not already heavily used
                    current_dosage = enhanced_dosages.get(fert.name, 0)
                    if current_dosage > 2.0:  # Already using a lot
                        efficiency *= 0.5
                    
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_fert = fert
            
            if best_fert:
                # Calculate required dosage
                cation_content = best_fert.composition.cations.get(micro, 0)
                anion_content = best_fert.composition.anions.get(micro, 0)
                total_content = cation_content + anion_content
                
                if total_content > 0:
                    # Calculate dosage needed (mg/L)
                    required_dosage_mg_l = (need * 100 * 100) / (total_content * best_fert.chemistry.purity)
                    required_dosage_g_l = required_dosage_mg_l / 1000
                    
                    # Apply reasonable limits
                    max_additional_dosage = min(required_dosage_g_l, 1.0)  # Max 1 g/L additional
                    
                    current_dosage = enhanced_dosages.get(best_fert.name, 0)
                    new_dosage = current_dosage + max_additional_dosage
                    
                    # Don't exceed 3 g/L per fertilizer
                    new_dosage = min(new_dosage, 3.0)
                    
                    enhanced_dosages[best_fert.name] = new_dosage
                    
                    print(f"  Added {max_additional_dosage:.3f} g/L of {best_fert.name} for {micro}")
        
        return enhanced_dosages

    def create_linear_programming_constraints(self, 
                                            fertilizers: List,
                                            targets: Dict[str, float],
                                            water: Dict[str, float],
                                            max_individual_dosage: float = 5.0,
                                            max_total_dosage: float = 15.0) -> Dict[str, Any]:
        """
        Create constraint matrices and bounds for linear programming optimization
        """
        
        n_fertilizers = len(fertilizers)
        nutrients = list(targets.keys())
        n_nutrients = len(nutrients)
        
        print(f"[LP] Creating constraints for {n_fertilizers} fertilizers and {n_nutrients} nutrients")
        
        # Build nutrient contribution matrix A
        # A[i,j] = contribution of fertilizer j to nutrient i per g/L dosage
        A_eq = np.zeros((n_nutrients, n_fertilizers))
        b_eq = np.zeros(n_nutrients)
        
        for i, nutrient in enumerate(nutrients):
            target_mg_l = targets[nutrient]
            water_contribution = water.get(nutrient, 0.0)
            
            # Right-hand side: target - water contribution
            b_eq[i] = target_mg_l - water_contribution
            
            for j, fert in enumerate(fertilizers):
                # Get nutrient content from fertilizer
                cation_content = fert.composition.cations.get(nutrient, 0.0)
                anion_content = fert.composition.anions.get(nutrient, 0.0)
                total_content = cation_content + anion_content
                
                if total_content > 0:
                    # Convert to mg/L contribution per g/L dosage
                    contribution_factor = (total_content * fert.chemistry.purity / 100.0 * 
                                        1000.0 / 100.0)
                    A_eq[i, j] = contribution_factor
        
        # Individual dosage bounds
        dosage_bounds = [(0, max_individual_dosage) for _ in range(n_fertilizers)]
        
        # Total dosage constraint
        # Sum of all dosages <= max_total_dosage
        A_ub = np.ones((1, n_fertilizers))
        b_ub = np.array([max_total_dosage])
        
        # Minimum significant dosage constraints (handled in post-processing)
        min_dosage_threshold = 0.001
        
        return {
            'A_eq': A_eq,                    # Equality constraint matrix (nutrient balance)
            'b_eq': b_eq,                    # Equality constraint RHS
            'A_ub': A_ub,                    # Inequality constraint matrix (total dosage)
            'b_ub': b_ub,                    # Inequality constraint RHS
            'bounds': dosage_bounds,         # Variable bounds
            'nutrients': nutrients,          # Nutrient order
            'fertilizers': [f.name for f in fertilizers],  # Fertilizer order
            'min_dosage_threshold': min_dosage_threshold,
            'constraint_summary': {
                'equality_constraints': n_nutrients,
                'inequality_constraints': 1,
                'variables': n_fertilizers,
                'max_individual_dosage': max_individual_dosage,
                'max_total_dosage': max_total_dosage
            }
        }

    def post_process_linear_programming_solution(self,
                                            raw_dosages: np.ndarray,
                                            fertilizer_names: List[str],
                                            min_threshold: float = 0.001) -> Dict[str, float]:
        """
        Post-process raw linear programming solution
        """
        
        print(f"[LP] Post-processing solution with {len(raw_dosages)} variables")
        
        # Convert to dictionary
        dosages = {}
        for i, fert_name in enumerate(fertilizer_names):
            raw_value = raw_dosages[i] if i < len(raw_dosages) else 0.0
            
            # Apply minimum threshold
            if raw_value >= min_threshold:
                dosages[fert_name] = raw_value
            else:
                dosages[fert_name] = 0.0
        
        # Remove zero dosages for cleaner output
        non_zero_dosages = {name: dosage for name, dosage in dosages.items() if dosage > 0}
        
        # Ensure all fertilizers are represented (for consistency)
        for fert_name in fertilizer_names:
            if fert_name not in dosages:
                dosages[fert_name] = 0.0
        
        active_count = len(non_zero_dosages)
        total_dosage = sum(dosages.values())
        
        print(f"[LP] Post-processing complete:")
        print(f"  Active fertilizers: {active_count}")
        print(f"  Total dosage: {total_dosage:.3f} g/L")
        print(f"  Cleaned {len(dosages) - active_count} negligible dosages")
        
        return dosages

    def generate_linear_programming_report(self,
                                        lp_result,
                                        targets: Dict[str, float],
                                        water: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive report for linear programming optimization results
        """
        
        print(f"[LP] Generating comprehensive optimization report...")
        
        # Performance analysis
        total_nutrients = len(lp_result.deviations_percent)
        excellent_count = len([d for d in lp_result.deviations_percent.values() if abs(d) <= 0.1])
        good_count = len([d for d in lp_result.deviations_percent.values() if 0.1 < abs(d) <= 5.0])
        acceptable_count = len([d for d in lp_result.deviations_percent.values() if 5.0 < abs(d) <= 15.0])
        poor_count = len([d for d in lp_result.deviations_percent.values() if abs(d) > 15.0])
        
        success_rate = (excellent_count + good_count) / total_nutrients * 100
        
        # Create detailed nutrient analysis
        nutrient_details = []
        for nutrient, deviation in lp_result.deviations_percent.items():
            target = targets.get(nutrient, 0)
            achieved = lp_result.achieved_concentrations.get(nutrient, 0)
            water_contrib = water.get(nutrient, 0)
            
            # Determine status and type
            if abs(deviation) <= 0.1:
                status = "Excellent"
            elif abs(deviation) <= 5.0:
                status = "Good"
            elif deviation < -15.0:
                status = "Deviation Low"
            elif deviation < 0:
                status = "Low"
            elif deviation > 15.0:
                status = "Deviation High"
            else:
                status = "High"
            
            nutrient_type = "Macro" if nutrient in ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'HCO3'] else "Micro"
            
            nutrient_details.append({
                'nutrient': nutrient,
                'target_mg_l': target,
                'achieved_mg_l': achieved,
                'water_contribution_mg_l': water_contrib,
                'fertilizer_contribution_mg_l': achieved - water_contrib,
                'deviation_percent': deviation,
                'status': status,
                'type': nutrient_type
            })
        
        # Sort by absolute deviation (best first)
        nutrient_details.sort(key=lambda x: abs(x['deviation_percent']))
        
        # Fertilizer efficiency analysis
        fertilizer_details = []
        for fert_name, dosage in lp_result.dosages_g_per_L.items():
            if dosage > 0:
                fertilizer_details.append({
                    'fertilizer': fert_name,
                    'dosage_g_per_L': dosage,
                    'dosage_ml_per_L': dosage,  # Assuming density = 1
                    'cost_efficiency': 'TBD',  # Could be calculated if cost data available
                    'utilization_percent': min(100, dosage / 3.0 * 100)  # Assuming 3g/L as full utilization
                })
        
        # Sort by dosage (highest first)
        fertilizer_details.sort(key=lambda x: x['dosage_g_per_L'], reverse=True)
        
        return {
            'optimization_summary': {
                'method': 'Linear Programming',
                'status': lp_result.optimization_status,
                'solver_time_seconds': lp_result.solver_time_seconds,
                'objective_value': lp_result.objective_value,
                'success_rate_percent': success_rate
            },
            'performance_metrics': {
                'total_nutrients': total_nutrients,
                'excellent_nutrients': excellent_count,
                'good_nutrients': good_count,
                'acceptable_nutrients': acceptable_count,
                'poor_nutrients': poor_count,
                'average_deviation_percent': np.mean(list(lp_result.deviations_percent.values())),
                'max_deviation_percent': max(abs(d) for d in lp_result.deviations_percent.values()),
                'ionic_balance_error_percent': lp_result.ionic_balance_error
            },
            'fertilizer_usage': {
                'active_fertilizers': lp_result.active_fertilizers,
                'total_dosage_g_per_L': lp_result.total_dosage,
                'average_dosage_per_fertilizer': lp_result.total_dosage / max(lp_result.active_fertilizers, 1),
                'max_individual_dosage': max(lp_result.dosages_g_per_L.values()) if lp_result.dosages_g_per_L else 0
            },
            'detailed_analysis': {
                'nutrient_details': nutrient_details,
                'fertilizer_details': fertilizer_details
            },
            'recommendations': self._generate_linear_programming_recommendations(lp_result, targets)
        }

    def _generate_linear_programming_recommendations(self, lp_result, targets: Dict[str, float]) -> List[str]:
        """
        Generate optimization recommendations based on LP results
        """
        
        recommendations = []
        
        # Performance-based recommendations
        avg_deviation = np.mean([abs(d) for d in lp_result.deviations_percent.values()])
        
        if lp_result.optimization_status == "Optimal":
            recommendations.append("✓ Linear programming found optimal solution")
        else:
            recommendations.append(f"⚠ Optimization status: {lp_result.optimization_status}")
        
        if avg_deviation <= 2.0:
            recommendations.append("✓ Excellent nutrient targeting achieved")
        elif avg_deviation <= 5.0:
            recommendations.append("✓ Good nutrient targeting achieved")
        elif avg_deviation <= 15.0:
            recommendations.append("⚠ Acceptable targeting, consider fertilizer selection review")
        else:
            recommendations.append("⚠ Poor targeting, fertilizer database may need expansion")
        
        # Ionic balance recommendations
        if lp_result.ionic_balance_error <= 5.0:
            recommendations.append("✓ Excellent ionic balance maintained")
        elif lp_result.ionic_balance_error <= 15.0:
            recommendations.append("✓ Good ionic balance achieved")
        else:
            recommendations.append("⚠ Ionic balance needs attention, consider pH adjustment")
        
        # Dosage recommendations
        if lp_result.total_dosage <= 8.0:
            recommendations.append("✓ Conservative dosage levels")
        elif lp_result.total_dosage <= 12.0:
            recommendations.append("✓ Moderate dosage levels")
        else:
            recommendations.append("⚠ High dosage levels, monitor for precipitation")
        
        # Fertilizer efficiency recommendations
        if lp_result.active_fertilizers <= 3:
            recommendations.append("✓ Efficient fertilizer usage (few sources)")
        elif lp_result.active_fertilizers <= 6:
            recommendations.append("✓ Balanced fertilizer usage")
        else:
            recommendations.append("⚠ Many fertilizers used, consider simplification")
        
        return recommendations

        
    def calculate_optimized_dosages(self, 
                                fertilizers: List,
                                target_concentrations: Dict[str, float],
                                water_analysis: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate optimized fertilizer dosages using strategic nutrient prioritization
        """
        print(f"[CALC] Calculating optimized dosages for {len(fertilizers)} fertilizers...")
        
        # Calculate remaining nutrients after water contribution
        remaining_nutrients = {}
        for element, target in target_concentrations.items():
            water_content = water_analysis.get(element, 0)
            remaining = max(0, target - water_content)
            remaining_nutrients[element] = remaining
            
            if remaining > 0:
                print(f"  [TARGET] {element}: Target={target:.3f}, Water={water_content:.3f}, Need={remaining:.3f} mg/L")

        results = {}
        
        # Step 2: Macronutrients first (existing logic)
        print(f"\n[CALC] STEP 1: MACRONUTRIENTS")
        
        # Phosphorus sources
        if remaining_nutrients.get('P', 0) > 0:
            p_fertilizers = [f for f in fertilizers if f.composition.anions.get('P', 0) > 5]
            if p_fertilizers:
                # Prefer monopotassium phosphate
                mkp_ferts = [f for f in p_fertilizers if 'monopotasic' in f.name.lower() or 'monopotásico' in f.name.lower()]
                best_p_fert = mkp_ferts[0] if mkp_ferts else p_fertilizers[0]
                
                p_needed = remaining_nutrients['P']
                dosage = self._calculate_dosage_for_element(best_p_fert, 'P', p_needed)
                
                if dosage > 0:
                    results[best_p_fert.name] = dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_p_fert, dosage)
                    print(f"    [SUCCESS] {best_p_fert.name}: {dosage/1000:.3f} g/L")
        
        # Calcium sources
        if remaining_nutrients.get('Ca', 0) > 0:
            ca_fertilizers = [f for f in fertilizers if f.composition.cations.get('Ca', 0) > 10]
            if ca_fertilizers:
                best_ca_fert = max(ca_fertilizers, key=lambda f: f.composition.cations.get('Ca', 0))
                ca_needed = remaining_nutrients['Ca']
                dosage = self._calculate_dosage_for_element(best_ca_fert, 'Ca', ca_needed)
                
                if dosage > 0:
                    results[best_ca_fert.name] = results.get(best_ca_fert.name, 0) + dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_ca_fert, dosage)
                    print(f"    [SUCCESS] {best_ca_fert.name}: {dosage/1000:.3f} g/L")
        
        # Potassium sources
        if remaining_nutrients.get('K', 0) > 0:
            k_fertilizers = [f for f in fertilizers if f.composition.cations.get('K', 0) > 20]
            if k_fertilizers:
                # Prefer KNO3 for balanced N-K supply
                kno3_ferts = [f for f in k_fertilizers if 'nitrato de potasio' in f.name.lower()]
                best_k_fert = kno3_ferts[0] if kno3_ferts else k_fertilizers[0]
                
                k_needed = remaining_nutrients['K']
                dosage = self._calculate_dosage_for_element(best_k_fert, 'K', k_needed)
                
                if dosage > 0:
                    results[best_k_fert.name] = results.get(best_k_fert.name, 0) + dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_k_fert, dosage)
                    print(f"    [SUCCESS] {best_k_fert.name}: {dosage/1000:.3f} g/L")
        
        # Magnesium sources
        if remaining_nutrients.get('Mg', 0) > 0:
            mg_fertilizers = [f for f in fertilizers if f.composition.cations.get('Mg', 0) > 5]
            if mg_fertilizers:
                best_mg_fert = max(mg_fertilizers, key=lambda f: f.composition.cations.get('Mg', 0))
                mg_needed = remaining_nutrients['Mg']
                dosage = self._calculate_dosage_for_element(best_mg_fert, 'Mg', mg_needed)
                
                if dosage > 0:
                    results[best_mg_fert.name] = results.get(best_mg_fert.name, 0) + dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_mg_fert, dosage)
                    print(f"    [SUCCESS] {best_mg_fert.name}: {dosage/1000:.3f} g/L")
        
        # Nitrogen sources (if still needed)
        if remaining_nutrients.get('N', 0) > 0:
            n_fertilizers = [f for f in fertilizers if (f.composition.cations.get('N', 0) + f.composition.anions.get('N', 0)) > 10]
            if n_fertilizers:
                # Prefer nitrate sources
                nitrate_ferts = [f for f in n_fertilizers if 'nitrato' in f.name.lower()]
                best_n_fert = nitrate_ferts[0] if nitrate_ferts else n_fertilizers[0]
                
                n_needed = remaining_nutrients['N']
                dosage = self._calculate_dosage_for_element(best_n_fert, 'N', n_needed)
                
                if dosage > 0:
                    results[best_n_fert.name] = results.get(best_n_fert.name, 0) + dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_n_fert, dosage)
                    print(f"    [SUCCESS] {best_n_fert.name}: {dosage/1000:.3f} g/L")
        
        # Sulfur sources (if still needed)
        if remaining_nutrients.get('S', 0) > 0:
            s_fertilizers = [f for f in fertilizers if f.composition.anions.get('S', 0) > 5]
            if s_fertilizers:
                best_s_fert = max(s_fertilizers, key=lambda f: f.composition.anions.get('S', 0))
                s_needed = remaining_nutrients['S']
                dosage = self._calculate_dosage_for_element(best_s_fert, 'S', s_needed)
                
                if dosage > 0:
                    results[best_s_fert.name] = results.get(best_s_fert.name, 0) + dosage / 1000.0
                    self._update_remaining_nutrients(remaining_nutrients, best_s_fert, dosage)
                    print(f"    [SUCCESS] {best_s_fert.name}: {dosage/1000:.3f} g/L")
        
        # Step 3: Micronutrients
        print(f"\n[CALC] STEP 2: MICRONUTRIENTS")
        micronutrients = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        for micro in micronutrients:
            if remaining_nutrients.get(micro, 0) > 0:
                micro_fertilizers = [f for f in fertilizers if 
                                (f.composition.cations.get(micro, 0) + f.composition.anions.get(micro, 0)) > 0.1]
                
                if micro_fertilizers:
                    best_micro_fert = max(micro_fertilizers, key=lambda f: 
                                        f.composition.cations.get(micro, 0) + f.composition.anions.get(micro, 0))
                    
                    micro_needed = remaining_nutrients[micro]
                    dosage = self._calculate_dosage_for_element(best_micro_fert, micro, micro_needed)
                    
                    if dosage > 0:
                        # Limit micronutrient fertilizer dosages
                        dosage_g_l = min(dosage / 1000.0, 1.0)  # Max 1 g/L for micronutrients
                        results[best_micro_fert.name] = results.get(best_micro_fert.name, 0) + dosage_g_l
                        print(f"    [SUCCESS] {best_micro_fert.name}: {dosage_g_l:.3f} g/L")
        
        # Fill with zeros for unused fertilizers
        for fert in fertilizers:
            if fert.name not in results:
                results[fert.name] = 0.0
        
        active_fertilizers = len([d for d in results.values() if d > 0])
        total_dosage = sum(results.values())
        
        print(f"\n[CALC] Optimization complete:")
        print(f"  Active fertilizers: {active_fertilizers}")
        print(f"  Total dosage: {total_dosage:.3f} g/L")
        
        return results

        def _calculate_dosage_for_element(self, fertilizer, element: str, needed_mg_l: float) -> float:
            """
            Calculate fertilizer dosage needed to supply a specific amount of an element
            """
            cation_content = fertilizer.composition.cations.get(element, 0)
            anion_content = fertilizer.composition.anions.get(element, 0)
            total_content = cation_content + anion_content
            
            if total_content <= 0:
                return 0.0
            
            # Calculate fertilizer amount needed (mg/L)
            # needed_mg_l = fertilizer_mg_l * (content% / 100) * (purity% / 100)
            # fertilizer_mg_l = needed_mg_l / (content% / 100) / (purity% / 100)
            fertilizer_mg_l = needed_mg_l * 100 * 100 / (total_content * fertilizer.chemistry.purity)
            
            return max(0, fertilizer_mg_l)

        def _update_remaining_nutrients(self, remaining_nutrients: Dict[str, float], 
                                    fertilizer, dosage_mg_l: float):
            """
            Update remaining nutrient needs after adding a fertilizer
            """
            # Update for cations
            for element, content_percent in fertilizer.composition.cations.items():
                if content_percent > 0:
                    contribution = self.calculate_element_contribution(
                        dosage_mg_l, content_percent, fertilizer.chemistry.purity
                    )
                    if element in remaining_nutrients:
                        remaining_nutrients[element] = max(0, remaining_nutrients[element] - contribution)
            
            # Update for anions
            for element, content_percent in fertilizer.composition.anions.items():
                if content_percent > 0:
                    contribution = self.calculate_element_contribution(
                        dosage_mg_l, content_percent, fertilizer.chemistry.purity
                    )
                    if element in remaining_nutrients:
                        remaining_nutrients[element] = max(0, remaining_nutrients[element] - contribution)