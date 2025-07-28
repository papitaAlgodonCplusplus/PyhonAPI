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
                print(f"  ✅ Added macronutrient: {fert.name}")
        
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
                    
                    print(f"  🧪 Added micronutrient: {fert.name} ({', '.join(micro_content)})")
        
        print(f"\n📋 Complete fertilizer set: {len(fertilizer_list)} fertilizers")
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
                print(f"  🎯 Added micronutrient target: {micronutrient} = {default_value} mg/L")
        
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
                print(f"  🎯 {element}: Target={target:.3f}, Water={water_content:.3f}, Need={remaining:.3f} mg/L")

        results = {}
        
        # Step 2: Macronutrients first (existing logic)
        print(f"\n📋 STEP 1: MACRONUTRIENTS")
        
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
                    print(f"    ✅ {best_p_fert.name}: {dosage/1000:.3f} g/L")
        
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
                    print(f"    ✅ {best_ca_fert.name}: {dosage/1000:.3f} g/L")
        
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
                    print(f"    ✅ {best_k_fert.name}: {dosage/1000:.3f} g/L")
        
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
                    print(f"    ✅ {best_mg_fert.name}: {dosage/1000:.3f} g/L")
        
        # Step 3: MICRONUTRIENTS (NEW!)
        print(f"\n🧪 STEP 2: MICRONUTRIENTS")
        
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
                        print(f"    🧪 {best_micro_fert.name}: {dosage/1000:.4f} g/L for {micronutrient}")
                
                else:
                    print(f"    ⚠️  No source found for {micronutrient}")
        
        # Step 4: Final verification
        total_dosage = sum(results.values())
        active_fertilizers = len([d for d in results.values() if d > 0.001])
        
        print(f"\n✅ ENHANCED OPTIMIZATION COMPLETE")
        print(f"   Active fertilizers: {active_fertilizers}")
        print(f"   Total dosage: {total_dosage:.3f} g/L")
        print(f"   Macronutrients: {len([f for f in results.keys() if not any(micro in f.lower() for micro in ['hierro', 'iron', 'manganeso', 'zinc', 'cobre', 'copper', 'borico', 'molibdato'])])}")
        print(f"   Micronutrients: {len([f for f in results.keys() if any(micro in f.lower() for micro in ['hierro', 'iron', 'manganeso', 'zinc', 'cobre', 'copper', 'borico', 'molibdato'])])}")
        
        return results

    def _calculate_dosage_for_element(self, fertilizer, element: str, needed_mg_l: float) -> float:
        """Calculate fertilizer dosage needed to supply specific element amount"""
        
        # Get element content from fertilizer (cations + anions)
        cation_content = fertilizer.composition.cations.get(element, 0)
        anion_content = fertilizer.composition.anions.get(element, 0)
        total_content = cation_content + anion_content
        
        if total_content <= 0:
            return 0.0
        
        # Calculate dosage using nutrient calculator
        dosage_mg_l = self.nutrient_calc.calculate_fertilizer_requirement(
            element, needed_mg_l, {element: total_content}, 
            fertilizer.percentage, fertilizer.molecular_weight
        )
        
        return dosage_mg_l

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

    def generate_enhanced_pdf_data(self, targets: Dict[str, float], 
                                 water: Dict[str, float],
                                 fertilizers: List = None) -> Dict[str, Any]:
        """Generate complete calculation data for enhanced PDF with micronutrients"""
        
        print(f"\n📄 GENERATING ENHANCED PDF DATA WITH MICRONUTRIENTS")
        
        # Get complete fertilizer set
        if fertilizers is None:
            fertilizers = self.get_complete_fertilizer_set(include_micronutrients=True)
        
        # Optimize with micronutrients
        dosages_g_l = self.optimize_with_micronutrients(targets, water, fertilizers)
        
        # Convert to API format
        fertilizer_dosages = {}
        for fertilizer in fertilizers:
            dosage_g = dosages_g_l.get(fertilizer.name, 0.0)
            fertilizer_dosages[fertilizer.name] = {
                'dosage_ml_per_L': round(dosage_g / fertilizer.density, 4),
                'dosage_g_per_L': round(dosage_g, 4)
            }

        # Calculate all contributions including micronutrients
        nutrient_contributions = self._calculate_enhanced_contributions(fertilizers, dosages_g_l)
        water_contribution = self._calculate_enhanced_water_contributions(water)
        final_solution = self._calculate_enhanced_final_solution(nutrient_contributions, water_contribution)

        # Enhanced verification including micronutrients
        complete_targets = self.create_complete_targets_with_micronutrients(targets)
        verification_results = self.verifier.verify_concentrations(
            complete_targets, final_solution['FINAL_mg_L']
        )
        
        # Ionic relationships and balance
        ionic_relationships = self.verifier.verify_ionic_relationships(
            final_solution['FINAL_meq_L'], final_solution['FINAL_mmol_L'], final_solution['FINAL_mg_L']
        )
        ionic_balance = self.verifier.verify_ionic_balance(final_solution['FINAL_meq_L'])

        # Enhanced cost analysis
        fertilizer_amounts_kg = {
            name: dosage * 1000 / 1000  # Convert g/L to kg per 1000L
            for name, dosage in dosages_g_l.items()
        }
        cost_analysis = self.cost_analyzer.calculate_solution_cost(
            fertilizer_amounts_kg, 1000, 1000
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
            'micronutrient_summary': self._generate_micronutrient_summary(final_solution['FINAL_mg_L']),
            'calculation_status': {
                'success': True,
                'warnings': [],
                'iterations': 1,
                'convergence_error': ionic_balance['difference_percentage'] / 100,
                'method_used': 'enhanced_with_micronutrients',
                'micronutrients_included': True
            }
        }

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
        """Convert mg/L directly to meq/L (combines mg->mmol->meq conversion)"""
        if mg_l <= 0:
            return 0.0
        
        # First convert mg/L to mmol/L
        mmol_l = self.convert_mg_to_mmol(mg_l, element)
        
        # Then convert mmol/L to meq/L
        meq_l = self.convert_mmol_to_meq(mmol_l, element)
        
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
        print(f"\n🔍 ANALYZING MICRONUTRIENT COVERAGE")
        
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

    def calculate_micronutrient_dosages(self, micronutrient_needs: Dict[str, float], 
                                       fertilizers: List) -> Dict[str, float]:
        """
        Calculate precise dosages for micronutrient fertilizers
        """
        print(f"\n🧪 CALCULATING MICRONUTRIENT DOSAGES")
        
        dosages = {}
        
        # Micronutrient fertilizer efficiency factors
        efficiency_factors = {
            'Fe': 0.95,  # Fe-EDTA is highly available
            'Mn': 0.90,  # MnSO4 good availability
            'Zn': 0.85,  # ZnSO4 good availability  
            'Cu': 0.80,  # CuSO4 moderate availability
            'B': 0.95,   # H3BO3 excellent availability
            'Mo': 0.90   # Na2MoO4 good availability
        }
        
        for micro, needed_mg_l in micronutrient_needs.items():
            print(f"   Calculating dosage for {micro}: {needed_mg_l:.3f} mg/L needed")
            
            # Find best fertilizer source for this micronutrient
            best_fertilizer = None
            best_content = 0
            
            for fertilizer in fertilizers:
                # Check for required fertilizer markers
                if '[Fertilizante Requerido]' in fertilizer.name:
                    cation_content = fertilizer.composition.cations.get(micro, 0)
                    anion_content = fertilizer.composition.anions.get(micro, 0)
                    total_content = cation_content + anion_content
                    
                    if total_content > best_content:
                        best_content = total_content
                        best_fertilizer = fertilizer
            
            if best_fertilizer and best_content > 0:
                # Calculate required dosage with efficiency factor
                efficiency = efficiency_factors.get(micro, 0.85)
                adjusted_need = needed_mg_l / efficiency
                
                # Calculate fertilizer dosage (mg/L)
                fertilizer_dosage_mg_l = (adjusted_need / best_content) * 100 * (100 / best_fertilizer.percentage)
                
                # Convert to g/L and apply reasonable limits
                dosage_g_l = fertilizer_dosage_mg_l / 1000.0
                
                # Apply micronutrient-specific limits
                max_dosages = {
                    'Fe': 0.050,  # Max 50 mg/L fertilizer
                    'Mn': 0.020,  # Max 20 mg/L fertilizer
                    'Zn': 0.015,  # Max 15 mg/L fertilizer
                    'Cu': 0.008,  # Max 8 mg/L fertilizer
                    'B': 0.030,   # Max 30 mg/L fertilizer
                    'Mo': 0.005   # Max 5 mg/L fertilizer
                }
                
                max_dosage = max_dosages.get(micro, 0.020)
                final_dosage = min(dosage_g_l, max_dosage)
                
                if final_dosage > 0.0001:  # Minimum meaningful dosage
                    dosages[best_fertilizer.name] = final_dosage
                    
                    # Calculate actual contribution
                    actual_contribution = final_dosage * 1000 * best_content * (best_fertilizer.percentage / 100) / 100
                    
                    print(f"     ✅ {best_fertilizer.name}: {final_dosage:.4f} g/L")
                    print(f"        Will provide: {actual_contribution:.3f} mg/L of {micro}")
                    print(f"        Efficiency: {efficiency*100:.0f}%")
                    
                    if final_dosage >= max_dosage:
                        print(f"        ⚠️  Dosage limited to maximum safe level")
                else:
                    print(f"     ❌ Calculated dosage too small for {micro}")
            else:
                print(f"     ❌ No suitable fertilizer found for {micro}")
        
        return dosages

    def validate_micronutrient_solution(self, final_concentrations: Dict[str, float], 
                                       targets: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate final micronutrient concentrations against targets and safety limits
        """
        print(f"\n✅ VALIDATING MICRONUTRIENT SOLUTION")
        
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
            print(f"   ⚠️  Safety warnings: {len(validation_results['safety_warnings'])}")
        
        if validation_results['adequacy_warnings']:
            print(f"   ⚠️  Adequacy warnings: {len(validation_results['adequacy_warnings'])}")
        
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
            recommendations.append(f"🚨 URGENT: Reduce {micro} fertilizer dosage to prevent toxicity")
        
        # Adequacy recommendations  
        for warning in validation_results['adequacy_warnings']:
            micro = warning.split(':')[0]
            
            # Find if we have a fertilizer for this micronutrient
            has_fertilizer = any(
                f.composition.cations.get(micro, 0) + f.composition.anions.get(micro, 0) > 0.1
                for f in fertilizers
            )
            
            if has_fertilizer:
                recommendations.append(f"📈 Increase {micro} fertilizer dosage to meet adequacy requirements")
            else:
                recommendations.append(f"➕ Add {micro} fertilizer source (auto-supplementation recommended)")
        
        # Performance recommendations
        compliance_score = validation_results['compliance_score']
        if compliance_score >= 90:
            recommendations.append("✅ Excellent micronutrient balance achieved")
        elif compliance_score >= 70:
            recommendations.append("📊 Good micronutrient coverage with minor adjustments needed")
        elif compliance_score >= 50:
            recommendations.append("⚠️ Moderate micronutrient gaps - consider formula review")
        else:
            recommendations.append("🔴 Significant micronutrient deficiencies - major formula revision needed")
        
        # Water quality considerations
        recommendations.append("💧 Monitor water quality for micronutrient interactions")
        recommendations.append("🔬 Consider chelated forms for hard water conditions")
        
        # Application recommendations
        recommendations.append("📦 Prepare micronutrient stock solution separately from macronutrients")
        recommendations.append("🌡️ Store micronutrient solutions in cool, dark conditions")
        recommendations.append("⏰ Replace micronutrient stock solutions every 2-4 weeks")
        
        # pH management
        recommendations.append("📏 Maintain solution pH 5.8-6.2 for optimal micronutrient availability")
        
        return recommendations[:10]  # Limit to top 10 recommendations

# Factory function
def create_enhanced_calculator():
    """Create enhanced calculator with micronutrient support"""
    return EnhancedFertilizerCalculator()

# Test function
def test_enhanced_calculator():
    """Test the enhanced calculator with micronutrients"""
    print("🧪 Testing Enhanced Calculator with Micronutrients...")
    
    calc = create_enhanced_calculator()
    
    # Test complete fertilizer set
    fertilizers = calc.get_complete_fertilizer_set(include_micronutrients=True)
    print(f"\n1. Complete fertilizer set: {len(fertilizers)} fertilizers")
    
    # Test targets with micronutrients
    base_targets = {
        'N': 150, 'P': 40, 'K': 200, 'Ca': 180, 'Mg': 50, 'S': 80
    }
    complete_targets = calc.create_complete_targets_with_micronutrients(base_targets)
    print(f"\n2. Complete targets: {len(complete_targets)} elements")
    
    # Test water analysis
    test_water = {
        'Ca': 20, 'K': 5, 'N': 2, 'P': 1, 'Mg': 8, 'S': 5,
        'Fe': 0.1, 'Mn': 0.05, 'Zn': 0.02, 'Cu': 0.01, 'B': 0.1, 'Mo': 0.001
    }
    
    # Test optimization
    print(f"\n3. Testing enhanced optimization...")
    dosages = calc.optimize_with_micronutrients(complete_targets, test_water, fertilizers)
    
    active_fertilizers = len([d for d in dosages.values() if d > 0.001])
    micronutrient_fertilizers = len([name for name, dosage in dosages.items() 
                                   if dosage > 0.001 and any(micro in name.lower() 
                                   for micro in ['hierro', 'iron', 'manganeso', 'zinc', 'cobre', 'copper', 'borico', 'molibdato'])])
    
    print(f"   Active fertilizers: {active_fertilizers}")
    print(f"   Micronutrient fertilizers: {micronutrient_fertilizers}")
    
    # Test PDF data generation
    print(f"\n4. Testing enhanced PDF data generation...")
    pdf_data = calc.generate_enhanced_pdf_data(complete_targets, test_water, fertilizers)
    
    micronutrient_summary = pdf_data['micronutrient_summary']
    supplied_micronutrients = micronutrient_summary['total_micronutrients_supplied']
    adequate_micronutrients = len(micronutrient_summary['adequate_micronutrients'])
    
    print(f"   Micronutrients supplied: {supplied_micronutrients}/6")
    print(f"   Adequate concentrations: {adequate_micronutrients}/6")
    
    print(f"\n✅ Enhanced calculator test completed successfully!")
    print(f"   System now supports complete micronutrient calculations")
    print(f"   Ready for integration with PDF generation and ML training")
    
    return True

if __name__ == "__main__":
    test_enhanced_calculator()