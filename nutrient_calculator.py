#!/usr/bin/env python3
"""
COMPLETE NUTRIENT CALCULATOR MODULE
Advanced nutrient calculations with linear algebra support
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math

class NutrientCalculator:
    """Advanced nutrient calculator with comprehensive algorithms"""

    def __init__(self):
        # Complete element data for conversions and calculations
        self.element_data = {
            # Major Cations
            'Ca': {'atomic_weight': 40.08, 'valence': 2, 'is_cation': True, 'common_forms': ['Ca2+']},
            'K': {'atomic_weight': 39.10, 'valence': 1, 'is_cation': True, 'common_forms': ['K+']},
            'Mg': {'atomic_weight': 24.31, 'valence': 2, 'is_cation': True, 'common_forms': ['Mg2+']},
            'Na': {'atomic_weight': 22.99, 'valence': 1, 'is_cation': True, 'common_forms': ['Na+']},
            'NH4': {'atomic_weight': 18.04, 'valence': 1, 'is_cation': True, 'common_forms': ['NH4+']},
            
            # Micronutrient Cations
            'Fe': {'atomic_weight': 55.85, 'valence': 2, 'is_cation': True, 'common_forms': ['Fe2+', 'Fe3+']},
            'Mn': {'atomic_weight': 54.94, 'valence': 2, 'is_cation': True, 'common_forms': ['Mn2+']},
            'Zn': {'atomic_weight': 65.38, 'valence': 2, 'is_cation': True, 'common_forms': ['Zn2+']},
            'Cu': {'atomic_weight': 63.55, 'valence': 2, 'is_cation': True, 'common_forms': ['Cu2+']},
            
            # Major Anions
            'NO3': {'atomic_weight': 62.00, 'valence': 1, 'is_cation': False, 'common_forms': ['NO3-']},
            'N': {'atomic_weight': 14.01, 'valence': 1, 'is_cation': False, 'common_forms': ['NO3-', 'NH4+']},
            'SO4': {'atomic_weight': 96.06, 'valence': 2, 'is_cation': False, 'common_forms': ['SO42-']},
            'S': {'atomic_weight': 32.06, 'valence': 2, 'is_cation': False, 'common_forms': ['SO42-']},
            'Cl': {'atomic_weight': 35.45, 'valence': 1, 'is_cation': False, 'common_forms': ['Cl-']},
            'H2PO4': {'atomic_weight': 96.99, 'valence': 1, 'is_cation': False, 'common_forms': ['H2PO4-']},
            'P': {'atomic_weight': 30.97, 'valence': 1, 'is_cation': False, 'common_forms': ['H2PO4-', 'HPO42-']},
            'HCO3': {'atomic_weight': 61.02, 'valence': 1, 'is_cation': False, 'common_forms': ['HCO3-']},
            
            # Micronutrient Anions
            'B': {'atomic_weight': 10.81, 'valence': 3, 'is_cation': False, 'common_forms': ['H3BO3', 'BO33-']},
            'Mo': {'atomic_weight': 95.96, 'valence': 6, 'is_cation': False, 'common_forms': ['MoO42-']}
        }

    def calculate_fertilizer_requirement(self, target_element: str, target_concentration: float,
                                       fertilizer_composition: Dict[str, float],
                                       purity: float, molecular_weight: float) -> float:
        """
        Calculate fertilizer requirement with enhanced precision
        """
        print(f"    🧮 Calculating requirement for {target_element}: target={target_concentration:.2f} mg/L")

        if target_element not in fertilizer_composition:
            print(f"    ❌ Element {target_element} not found in composition")
            return 0.0

        element_weight_percent = fertilizer_composition[target_element]
        if element_weight_percent <= 0:
            print(f"    ❌ Element {target_element} has zero content: {element_weight_percent}")
            return 0.0

        # Enhanced calculation with purity adjustment
        effective_content = element_weight_percent * (purity / 100.0)
        
        if effective_content <= 0:
            print(f"    ❌ Effective content is zero after purity adjustment")
            return 0.0

        # Calculate fertilizer amount in mg/L
        fertilizer_amount = (target_concentration * 100.0) / effective_content

        # Validate result
        if fertilizer_amount < 0 or fertilizer_amount > 100000:  # Sanity check
            print(f"    ⚠️  Calculated amount seems unrealistic: {fertilizer_amount:.3f} mg/L")
            return 0.0

        print(f"    ✅ Calculated: {fertilizer_amount:.3f} mg/L fertilizer needed")
        print(f"       Element content: {element_weight_percent:.2f}%")
        print(f"       Effective content: {effective_content:.2f}%")
        
        return max(0, fertilizer_amount)

    def calculate_element_contribution(self, fertilizer_amount: float, element_weight_percent: float,
                                     purity: float) -> float:
        """
        Calculate element contribution from fertilizer amount with validation
        """
        if fertilizer_amount <= 0 or element_weight_percent <= 0:
            return 0.0

        # Calculate contribution with purity adjustment
        contribution = fertilizer_amount * element_weight_percent * (purity / 100.0) / 100.0
        
        # Validate result
        if contribution < 0:
            return 0.0
        
        return contribution

    def convert_mg_to_mmol(self, mg_l: float, element: str) -> float:
        """
        Convert mg/L to mmol/L with enhanced element support
        """
        if element not in self.element_data or mg_l <= 0:
            return 0.0
        
        atomic_weight = self.element_data[element]['atomic_weight']
        mmol_l = mg_l / atomic_weight
        
        return mmol_l

    def convert_mmol_to_meq(self, mmol_l: float, element: str) -> float:
        """
        Convert mmol/L to meq/L with enhanced element support
        """
        if element not in self.element_data or mmol_l <= 0:
            return 0.0
        
        valence = self.element_data[element]['valence']
        meq_l = mmol_l * valence
        
        return meq_l

    def convert_mg_to_meq_direct(self, mg_l: float, element: str) -> float:
        """
        Direct conversion from mg/L to meq/L
        """
        if element not in self.element_data or mg_l <= 0:
            return 0.0
        
        atomic_weight = self.element_data[element]['atomic_weight']
        valence = self.element_data[element]['valence']
        
        meq_l = (mg_l * valence) / atomic_weight
        
        return meq_l

    def calculate_ionic_strength(self, final_meq: Dict[str, float]) -> float:
        """
        Calculate ionic strength of the solution
        """
        ionic_strength = 0.0
        
        for element, meq_l in final_meq.items():
            if element in self.element_data and meq_l > 0:
                valence = self.element_data[element]['valence']
                # I = 0.5 * Σ(ci * zi^2) where ci is molarity and zi is charge
                molarity = meq_l / 1000  # Convert meq/L to eq/L, then to M
                ionic_strength += 0.5 * molarity * (valence ** 2)
        
        return ionic_strength

    def calculate_ec_advanced(self, final_meq: Dict[str, float]) -> float:
        """
        Advanced EC calculation using ionic contributions
        """
        # Define specific conductivity coefficients for different ions (mS/cm per meq/L)
        conductivity_coefficients = {
            'Ca': 0.060,
            'K': 0.074,
            'Mg': 0.053,
            'Na': 0.050,
            'NH4': 0.074,
            'Fe': 0.054,
            'Mn': 0.053,
            'Zn': 0.053,
            'Cu': 0.054,
            'N': 0.071,  # NO3-
            'S': 0.080,  # SO4^2-
            'Cl': 0.076,
            'P': 0.069,  # H2PO4-
            'HCO3': 0.045,
            'B': 0.040,
            'Mo': 0.075
        }
        
        total_conductivity = 0.0
        
        for element, meq_l in final_meq.items():
            if meq_l > 0 and element in conductivity_coefficients:
                conductivity = meq_l * conductivity_coefficients[element]
                total_conductivity += conductivity
        
        # Convert from mS/cm to dS/m
        ec_ds_m = total_conductivity / 10
        
        return ec_ds_m

    def calculate_ph_advanced(self, final_mg: Dict[str, float], final_meq: Dict[str, float]) -> float:
        """
        Advanced pH calculation considering buffer systems
        """
        # Get key parameters
        hco3_mg = final_mg.get('HCO3', 0)
        no3_mg = final_mg.get('N', 0)
        nh4_meq = final_meq.get('NH4', 0)
        h2po4_mg = final_mg.get('P', 0)
        
        # Base pH calculation
        base_ph = 6.0
        
        # HCO3 effect (alkalizing)
        if hco3_mg > 30:
            hco3_effect = (hco3_mg - 30) / 100 * 0.8
            base_ph += hco3_effect
        
        # NO3 effect (slight acidifying when high)
        if no3_mg > 150:
            no3_effect = (no3_mg - 150) / 200 * 0.3
            base_ph -= no3_effect
        
        # NH4 effect (acidifying)
        if nh4_meq > 1.0:
            nh4_effect = nh4_meq * 0.2
            base_ph -= nh4_effect
        
        # H2PO4 effect (buffering around 6.2)
        if h2po4_mg > 20:
            # Phosphate buffer pulls pH toward 6.2
            buffer_strength = min(h2po4_mg / 100, 0.5)
            base_ph = base_ph * (1 - buffer_strength) + 6.2 * buffer_strength
        
        # Ensure reasonable pH range
        calculated_ph = max(4.0, min(8.5, base_ph))
        
        return calculated_ph

    def calculate_osmotic_pressure(self, final_mg: Dict[str, float]) -> float:
        """
        Calculate osmotic pressure of the solution
        """
        # Van't Hoff equation: π = iMRT
        # Simplified calculation for practical use
        
        total_dissolved_solids = sum(final_mg.values())  # mg/L
        
        # Convert to approximate molarity
        average_molecular_weight = 60  # Approximate average for nutrient salts
        total_molarity = (total_dissolved_solids / 1000) / average_molecular_weight
        
        # Van't Hoff factor (average dissociation)
        van_hoff_factor = 2.5  # Average for typical fertilizer salts
        
        # Osmotic pressure in atm (R = 0.0821 L⋅atm/(mol⋅K), T = 298K)
        osmotic_pressure = van_hoff_factor * total_molarity * 0.0821 * 298
        
        return osmotic_pressure

    def prepare_linear_algebra_system(self, fertilizers: List, targets: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare linear algebra system for optimization: A * x = b
        Where A is composition matrix, x is dosages, b is targets
        """
        print(f"\n🔢 PREPARING LINEAR ALGEBRA SYSTEM")
        print(f"Fertilizers: {len(fertilizers)}")
        print(f"Target elements: {list(targets.keys())}")
        
        # Get all elements that appear in targets or fertilizers
        all_elements = set(targets.keys())
        for fertilizer in fertilizers:
            all_elements.update(fertilizer.composition.cations.keys())
            all_elements.update(fertilizer.composition.anions.keys())
        
        # Filter to only elements with targets or significant content
        relevant_elements = []
        for element in all_elements:
            if element in targets and targets[element] > 0:
                relevant_elements.append(element)
            elif any(
                fertilizer.composition.cations.get(element, 0) + 
                fertilizer.composition.anions.get(element, 0) > 1
                for fertilizer in fertilizers
            ):
                relevant_elements.append(element)
        
        relevant_elements = sorted(relevant_elements)
        print(f"Relevant elements: {relevant_elements}")
        
        # Create composition matrix A
        n_fertilizers = len(fertilizers)
        n_elements = len(relevant_elements)
        
        A = np.zeros((n_elements, n_fertilizers))
        
        for j, fertilizer in enumerate(fertilizers):
            for i, element in enumerate(relevant_elements):
                # Get element content from fertilizer (cations + anions)
                cation_content = fertilizer.composition.cations.get(element, 0)
                anion_content = fertilizer.composition.anions.get(element, 0)
                total_content = cation_content + anion_content
                
                # Convert to contribution per mg/L of fertilizer
                # content_percent * purity / 100 / 100 = fraction
                if total_content > 0:
                    contribution_factor = total_content * (fertilizer.percentage / 100.0) / 100.0
                    A[i, j] = contribution_factor
        
        # Create target vector b
        b = np.zeros(n_elements)
        for i, element in enumerate(relevant_elements):
            b[i] = targets.get(element, 0)
        
        print(f"System matrix A shape: {A.shape}")
        print(f"Target vector b shape: {b.shape}")
        print(f"Matrix condition number: {np.linalg.cond(A):.2e}")
        
        return A, b, relevant_elements

    def solve_linear_system(self, A: np.ndarray, b: np.ndarray, method: str = "least_squares") -> np.ndarray:
        """
        Solve linear system A * x = b using specified method
        """
        print(f"🔧 Solving linear system using {method}")
        
        try:
            if method == "least_squares":
                # Least squares solution (handles overdetermined systems)
                x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                print(f"Least squares residuals: {residuals}")
                print(f"Matrix rank: {rank}")
                
            elif method == "pseudoinverse":
                # Moore-Penrose pseudoinverse
                A_pinv = np.linalg.pinv(A)
                x = A_pinv @ b
                residual = np.linalg.norm(A @ x - b)
                print(f"Pseudoinverse residual: {residual:.6f}")
                
            elif method == "normal_equations":
                # Normal equations: (A^T A) x = A^T b
                ATA = A.T @ A
                ATb = A.T @ b
                
                # Add regularization for numerical stability
                regularization = 1e-8 * np.eye(ATA.shape[0])
                ATA_reg = ATA + regularization
                
                x = np.linalg.solve(ATA_reg, ATb)
                residual = np.linalg.norm(A @ x - b)
                print(f"Normal equations residual: {residual:.6f}")
                
            elif method == "non_negative_least_squares":
                # Non-negative least squares (requires scipy)
                try:
                    from scipy.optimize import nnls
                    x, residual = nnls(A, b)
                    print(f"NNLS residual: {residual:.6f}")
                except ImportError:
                    print("Scipy not available, falling back to least squares")
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    # Project negative values to zero
                    x = np.maximum(x, 0)
                    
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Validate solution
            achieved = A @ x
            print(f"Solution vector shape: {x.shape}")
            print(f"Solution norm: {np.linalg.norm(x):.6f}")
            print(f"Non-negative values: {np.sum(x >= 0)}/{len(x)}")
            
            return x
            
        except np.linalg.LinAlgError as e:
            print(f"❌ Linear algebra error: {e}")
            # Return zero solution
            return np.zeros(A.shape[1])
        except Exception as e:
            print(f"❌ Unexpected error in linear solver: {e}")
            return np.zeros(A.shape[1])

    def optimize_with_constraints(self, A: np.ndarray, b: np.ndarray, 
                                bounds: List[Tuple[float, float]] = None,
                                max_dosage: float = 10.0) -> np.ndarray:
        """
        Solve optimization problem with constraints using scipy.optimize
        """
        try:
            from scipy.optimize import minimize
            
            n_vars = A.shape[1]
            
            # Default bounds: non-negative with reasonable upper limit
            if bounds is None:
                bounds = [(0, max_dosage) for _ in range(n_vars)]
            
            # Objective function: minimize ||Ax - b||^2
            def objective(x):
                residual = A @ x - b
                return np.sum(residual ** 2)
            
            # Initial guess
            x0 = np.ones(n_vars) * 0.1
            
            # Solve optimization problem
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            if result.success:
                print(f"✅ Constrained optimization successful")
                print(f"Final objective value: {result.fun:.6f}")
                return result.x
            else:
                print(f"⚠️  Constrained optimization failed: {result.message}")
                # Fallback to least squares
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                return np.maximum(x, 0)  # Project to non-negative
                
        except ImportError:
            print("Scipy not available, using least squares with projection")
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return np.maximum(x, 0)  # Project to non-negative

    def calculate_nutrient_efficiency(self, achieved: Dict[str, float], 
                                    targets: Dict[str, float],
                                    fertilizer_costs: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate nutrient use efficiency and cost effectiveness
        """
        efficiency_metrics = {
            'nutrient_efficiency': {},
            'overall_efficiency': 0.0,
            'cost_effectiveness': 0.0,
            'waste_analysis': {}
        }
        
        total_efficiency = 0.0
        count = 0
        
        for element, target in targets.items():
            if target > 0 and element in achieved:
                actual = achieved[element]
                
                # Calculate efficiency (0-100%)
                if actual <= target:
                    efficiency = (actual / target) * 100
                else:
                    # Penalize over-application
                    excess = actual - target
                    penalty = min(excess / target, 1.0) * 50  # Up to 50% penalty
                    efficiency = max(0, 100 - penalty)
                
                efficiency_metrics['nutrient_efficiency'][element] = round(efficiency, 1)
                
                # Calculate waste
                if actual > target:
                    waste_amount = actual - target
                    waste_percentage = (waste_amount / target) * 100
                    efficiency_metrics['waste_analysis'][element] = {
                        'waste_amount': round(waste_amount, 2),
                        'waste_percentage': round(waste_percentage, 1)
                    }
                
                total_efficiency += efficiency
                count += 1
        
        # Calculate overall efficiency
        if count > 0:
            efficiency_metrics['overall_efficiency'] = round(total_efficiency / count, 1)
        
        # Calculate cost effectiveness (efficiency per dollar)
        total_cost = sum(fertilizer_costs.values())
        if total_cost > 0:
            efficiency_metrics['cost_effectiveness'] = round(
                efficiency_metrics['overall_efficiency'] / total_cost, 2
            )
        
        return efficiency_metrics

    def validate_solution_quality(self, final_mg: Dict[str, float], 
                                final_meq: Dict[str, float]) -> Dict[str, Any]:
        """
        Comprehensive solution quality validation
        """
        quality_report = {
            'overall_score': 0,
            'concentration_check': {},
            'ionic_balance_check': {},
            'ec_ph_check': {},
            'recommendations': []
        }
        
        # Check individual concentrations
        concentration_issues = []
        for element, concentration in final_mg.items():
            if element in self.element_data:
                # Check for extremely high concentrations
                if concentration > 1000:  # Very high for any element
                    concentration_issues.append(f"{element} is extremely high ({concentration:.1f} mg/L)")
                elif concentration > 500 and element in ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']:
                    concentration_issues.append(f"Micronutrient {element} is toxic level ({concentration:.1f} mg/L)")
        
        quality_report['concentration_check'] = {
            'issues_found': len(concentration_issues),
            'issues': concentration_issues
        }
        
        # Check ionic balance
        cations = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'Fe', 'Mn', 'Zn', 'Cu']
        anions = ['N', 'S', 'Cl', 'P', 'HCO3', 'B', 'Mo']
        
        cation_sum = sum(final_meq.get(cation, 0) for cation in cations)
        anion_sum = sum(final_meq.get(anion, 0) for anion in anions)
        
        if cation_sum > 0:
            balance_error = abs(cation_sum - anion_sum) / cation_sum * 100
        else:
            balance_error = 0
        
        quality_report['ionic_balance_check'] = {
            'cation_sum': round(cation_sum, 2),
            'anion_sum': round(anion_sum, 2),
            'balance_error_percent': round(balance_error, 1),
            'status': 'Good' if balance_error <= 10 else 'Poor'
        }
        
        # Check EC and pH
        calculated_ec = self.calculate_ec_advanced(final_meq)
        calculated_ph = self.calculate_ph_advanced(final_mg, final_meq)
        
        ec_status = 'Good'
        if calculated_ec < 1.0:
            ec_status = 'Low'
        elif calculated_ec > 3.0:
            ec_status = 'High'
        
        ph_status = 'Good'
        if calculated_ph < 5.5:
            ph_status = 'Low'
        elif calculated_ph > 7.0:
            ph_status = 'High'
        
        quality_report['ec_ph_check'] = {
            'calculated_ec': round(calculated_ec, 2),
            'calculated_ph': round(calculated_ph, 1),
            'ec_status': ec_status,
            'ph_status': ph_status
        }
        
        # Generate recommendations
        recommendations = []
        
        if concentration_issues:
            recommendations.append("Review fertilizer dosages - some concentrations are excessive")
        
        if balance_error > 15:
            recommendations.append("Significant ionic imbalance detected - adjust fertilizer ratios")
        
        if ec_status == 'High':
            recommendations.append("EC is high - consider dilution or reduced fertilizer concentration")
        elif ec_status == 'Low':
            recommendations.append("EC is low - increase fertilizer concentration")
        
        if ph_status != 'Good':
            recommendations.append(f"pH is {ph_status.lower()} - consider pH adjustment")
        
        if not recommendations:
            recommendations.append("Solution quality is acceptable")
        
        quality_report['recommendations'] = recommendations
        
        # Calculate overall score (0-100)
        score = 100
        
        # Deduct for concentration issues
        score -= len(concentration_issues) * 10
        
        # Deduct for ionic balance
        if balance_error > 10:
            score -= min(balance_error - 10, 30)
        
        # Deduct for EC/pH issues
        if ec_status != 'Good':
            score -= 10
        if ph_status != 'Good':
            score -= 10
        
        quality_report['overall_score'] = max(0, round(score))
        
        return quality_report

    def get_element_info(self, element: str) -> Dict[str, Any]:
        """
        Get comprehensive information about an element
        """
        if element not in self.element_data:
            return {'error': f'Element {element} not found in database'}
        
        data = self.element_data[element]
        
        return {
            'element': element,
            'atomic_weight': data['atomic_weight'],
            'valence': data['valence'],
            'is_cation': data['is_cation'],
            'common_forms': data['common_forms'],
            'conversion_factors': {
                'mg_to_mmol': 1 / data['atomic_weight'],
                'mmol_to_meq': data['valence'],
                'mg_to_meq': data['valence'] / data['atomic_weight']
            }
        }

    def batch_convert_units(self, concentrations: Dict[str, float], 
                          from_unit: str, to_unit: str) -> Dict[str, float]:
        """
        Batch convert concentrations between different units
        """
        valid_units = ['mg/L', 'mmol/L', 'meq/L', 'ppm']
        
        if from_unit not in valid_units or to_unit not in valid_units:
            raise ValueError(f"Units must be one of: {valid_units}")
        
        converted = {}
        
        for element, concentration in concentrations.items():
            if element in self.element_data and concentration > 0:
                
                # Convert to mg/L first (common base)
                if from_unit == 'mg/L' or from_unit == 'ppm':
                    mg_l = concentration
                elif from_unit == 'mmol/L':
                    mg_l = concentration * self.element_data[element]['atomic_weight']
                elif from_unit == 'meq/L':
                    atomic_weight = self.element_data[element]['atomic_weight']
                    valence = self.element_data[element]['valence']
                    mg_l = concentration * atomic_weight / valence
                
                # Convert from mg/L to target unit
                if to_unit == 'mg/L' or to_unit == 'ppm':
                    converted[element] = mg_l
                elif to_unit == 'mmol/L':
                    converted[element] = mg_l / self.element_data[element]['atomic_weight']
                elif to_unit == 'meq/L':
                    atomic_weight = self.element_data[element]['atomic_weight']
                    valence = self.element_data[element]['valence']
                    converted[element] = (mg_l * valence) / atomic_weight
            else:
                converted[element] = 0.0
        
        return converted


# Helper functions for standalone usage
def test_nutrient_calculator():
    """
    Test function to verify nutrient calculator functionality
    """
    print("🧪 Testing Nutrient Calculator...")
    
    calc = NutrientCalculator()
    
    # Test 1: Basic calculations
    print("\n1. Basic Calculations:")
    
    target_mg = 150  # mg/L N
    composition = {'N': 13.85}  # KNO3
    purity = 98.0
    mw = 101.1
    
    required = calc.calculate_fertilizer_requirement('N', target_mg, composition, purity, mw)
    print(f"   Required KNO3: {required:.2f} mg/L")
    
    contribution = calc.calculate_element_contribution(required, 13.85, 98.0)
    print(f"   N contribution: {contribution:.2f} mg/L")
    
    # Test 2: Unit conversions
    print("\n2. Unit Conversions:")
    
    test_concentrations = {'Ca': 180, 'K': 200, 'N': 150}
    
    mmol_values = {}
    meq_values = {}
    
    for element, mg_l in test_concentrations.items():
        mmol = calc.convert_mg_to_mmol(mg_l, element)
        meq = calc.convert_mmol_to_meq(mmol, element)
        mmol_values[element] = mmol
        meq_values[element] = meq
        print(f"   {element}: {mg_l} mg/L = {mmol:.2f} mmol/L = {meq:.2f} meq/L")
    
    # Test 3: Advanced calculations
    print("\n3. Advanced Calculations:")
    
    ec = calc.calculate_ec_advanced(meq_values)
    print(f"   Calculated EC: {ec:.2f} dS/m")
    
    ph = calc.calculate_ph_advanced(test_concentrations, meq_values)
    print(f"   Calculated pH: {ph:.1f}")
    
    ionic_strength = calc.calculate_ionic_strength(meq_values)
    print(f"   Ionic strength: {ionic_strength:.4f} M")
    
    # Test 4: Linear algebra system
    print("\n4. Linear Algebra Test:")
    print("   (Requires fertilizer objects - skipped in basic test)")
    
    print("\n✅ Nutrient calculator test completed!")
    return True


if __name__ == "__main__":
    # Run test if executed directly
    test_nutrient_calculator()