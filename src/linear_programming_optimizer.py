#!/usr/bin/env python3
"""
LINEAR PROGRAMMING FERTILIZER OPTIMIZER
Advanced optimization using PuLP/SciPy to achieve 0% deviation from targets
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

# Linear Programming imports with fallbacks
try:
    import pulp as lp
    PULP_AVAILABLE = True
    print("[LP] PuLP linear programming solver available")
except ImportError:
    PULP_AVAILABLE = False
    print("[LP] PuLP not available, falling back to SciPy")

try:
    from scipy.optimize import linprog, minimize, Bounds, LinearConstraint
    SCIPY_AVAILABLE = True
    print("[LP] SciPy optimization available")
except ImportError:
    SCIPY_AVAILABLE = False
    print("[LP] SciPy not available")

from nutrient_calculator import EnhancedFertilizerCalculator

@dataclass
class LinearProgrammingResult:
    """Results from linear programming optimization"""
    dosages_g_per_L: Dict[str, float]
    achieved_concentrations: Dict[str, float]
    deviations_percent: Dict[str, float]
    optimization_status: str
    objective_value: float
    ionic_balance_error: float
    solver_time_seconds: float
    active_fertilizers: int
    total_dosage: float

class LinearProgrammingOptimizer:
    """Advanced linear programming optimizer for fertilizer calculations"""
    
    def __init__(self):
        self.nutrient_calc = EnhancedFertilizerCalculator()
        
        # Priority weights for different objectives
        self.objective_weights = {
            'deviation_minimization': 1000.0,  # Highest priority: minimize deviations
            'cost_minimization': 1.0,          # Lower priority: minimize cost
            'dosage_minimization': 10.0,       # Medium priority: minimize total dosage
            'ionic_balance': 500.0             # High priority: maintain ionic balance
        }
        
        # Constraint limits
        self.max_individual_dosage = 5.0  # g/L
        self.max_total_dosage = 15.0      # g/L
        self.max_deviation_acceptable = 0.05  # 5% maximum acceptable deviation
        
    def optimize_fertilizer_solution(self, 
                                   fertilizers: List,
                                   target_concentrations: Dict[str, float],
                                   water_analysis: Dict[str, float],
                                   volume_liters: float = 1000,
                                   apply_safety_caps: bool = True,
                                   strict_caps: bool = True) -> LinearProgrammingResult:
        """
        Optimize fertilizer solution using linear programming to achieve 0% deviation
        """
        start_time = time.time()
        
        print(f"\n=== LINEAR PROGRAMMING OPTIMIZATION ===")
        print(f"Fertilizers: {len(fertilizers)}")
        print(f"Targets: {len(target_concentrations)}")
        print(f"Solver priority: {'PuLP' if PULP_AVAILABLE else 'SciPy'}")
        
        # Apply safety caps if requested
        safe_targets = target_concentrations.copy()
        if apply_safety_caps:
            from nutrient_caps import apply_nutrient_caps_to_targets
            caps_result = apply_nutrient_caps_to_targets(target_concentrations, strict_caps)
            safe_targets = caps_result['capped_concentrations']
            print(f"Safety caps applied: {caps_result['total_adjustments']} adjustments")
        
        # Try PuLP first, then SciPy
        if PULP_AVAILABLE:
            result = self._optimize_with_pulp(fertilizers, safe_targets, water_analysis, volume_liters)
        elif SCIPY_AVAILABLE:
            result = self._optimize_with_scipy(fertilizers, safe_targets, water_analysis, volume_liters)
        else:
            raise ImportError("Neither PuLP nor SciPy available for linear programming")
        
        result.solver_time_seconds = time.time() - start_time
        
        print(f"=== OPTIMIZATION COMPLETE ===")
        print(f"Status: {result.optimization_status}")
        print(f"Active fertilizers: {result.active_fertilizers}")
        print(f"Total dosage: {result.total_dosage:.3f} g/L")
        print(f"Solver time: {result.solver_time_seconds:.2f}s")
        print(f"Average deviation: {np.mean(list(result.deviations_percent.values())):.2f}%")
        
        return result
    
    def _optimize_with_pulp(self, 
                           fertilizers: List,
                           targets: Dict[str, float],
                           water: Dict[str, float],
                           volume_liters: float) -> LinearProgrammingResult:
        """
        Optimize using PuLP linear programming solver
        """
        print(f"[LP] Using PuLP solver for optimization...")
        
        # Create the problem
        prob = lp.LpProblem("Fertilizer_Optimization", lp.LpMinimize)
        
        # Decision variables: dosage of each fertilizer (g/L)
        dosage_vars = {}
        for fert in fertilizers:
            var_name = f"dosage_{fert.name.replace(' ', '_').replace('(', '').replace(')', '')}"
            dosage_vars[fert.name] = lp.LpVariable(var_name, lowBound=0, upBound=self.max_individual_dosage)
        
        # Deviation variables for each nutrient (both positive and negative)
        deviation_vars_pos = {}
        deviation_vars_neg = {}
        for nutrient in targets.keys():
            var_name = nutrient.replace(' ', '_')
            deviation_vars_pos[nutrient] = lp.LpVariable(f"dev_pos_{var_name}", lowBound=0)
            deviation_vars_neg[nutrient] = lp.LpVariable(f"dev_neg_{var_name}", lowBound=0)
        
        # Objective function: minimize weighted deviations and dosages
        objective = 0
        
        # 1. Minimize nutrient deviations (highest priority)
        for nutrient in targets.keys():
            target_value = targets[nutrient]
            if target_value > 0:
                weight = self.objective_weights['deviation_minimization'] / target_value
                objective += weight * (deviation_vars_pos[nutrient] + deviation_vars_neg[nutrient])
        
        # 2. Minimize total dosage (lower priority)
        for fert_name, var in dosage_vars.items():
            objective += self.objective_weights['dosage_minimization'] * var
        
        prob += objective
        
        # Constraints
        
        # 1. Nutrient balance constraints with deviations
        for nutrient in targets.keys():
            target_mg_l = targets[nutrient]
            water_contribution = water.get(nutrient, 0.0)
            
            # Calculate total contribution from fertilizers
            fertilizer_contribution = 0
            for fert in fertilizers:
                dosage_var = dosage_vars[fert.name]
                
                # Get nutrient content from fertilizer
                cation_content = fert.composition.cations.get(nutrient, 0.0)
                anion_content = fert.composition.anions.get(nutrient, 0.0)
                total_content = cation_content + anion_content
                
                if total_content > 0:
                    # Convert dosage to contribution (g/L * content% * purity% / 100 * 1000 mg/g)
                    contribution_factor = total_content * fert.chemistry.purity / 100.0 * 1000.0 / 100.0
                    fertilizer_contribution += dosage_var * contribution_factor
            
            # Total achieved = water + fertilizer contributions
            total_achieved = water_contribution + fertilizer_contribution
            
            # Balance equation with deviations
            # total_achieved = target + deviation_pos - deviation_neg
            prob += (total_achieved == target_mg_l + deviation_vars_pos[nutrient] - deviation_vars_neg[nutrient])
        
        # 2. Maximum dosage constraints
        for var in dosage_vars.values():
            prob += var <= self.max_individual_dosage
        
        # 3. Maximum total dosage constraint
        prob += lp.lpSum(dosage_vars.values()) <= self.max_total_dosage
        
        # 4. Minimum significant dosage (avoid tiny amounts)
        for var in dosage_vars.values():
            # If used, use at least 0.001 g/L (will be handled in post-processing)
            pass
        
        # Solve the problem
        print(f"[LP] Solving linear program with {len(dosage_vars)} variables...")
        
        # Try multiple solvers for robustness
        solvers_to_try = [lp.PULP_CBC_CMD, lp.GLPK_CMD, lp.CPLEX_CMD]
        
        solution_found = False
        for solver_class in solvers_to_try:
            try:
                solver = solver_class(msg=0)  # Silent mode
                prob.solve(solver)
                
                if prob.status == lp.LpStatusOptimal:
                    solution_found = True
                    print(f"[LP] Optimal solution found using {solver_class.__name__}")
                    break
                else:
                    print(f"[LP] {solver_class.__name__} failed: {lp.LpStatus[prob.status]}")
            except Exception as e:
                print(f"[LP] {solver_class.__name__} error: {e}")
                continue
        
        if not solution_found:
            print(f"[LP] All solvers failed, trying default solver...")
            prob.solve()
            solution_found = prob.status == lp.LpStatusOptimal
        
        # Extract results
        if solution_found:
            print(f"[LP] Optimization successful! Status: {lp.LpStatus[prob.status]}")
            
            # Extract dosages
            dosages = {}
            for fert_name, var in dosage_vars.items():
                dosage_value = var.varValue if var.varValue is not None else 0.0
                dosages[fert_name] = max(0.0, dosage_value)
            
            # Calculate achieved concentrations and deviations
            achieved_concentrations = self._calculate_achieved_concentrations(
                dosages, water, fertilizers
            )
            
            deviations_percent = {}
            for nutrient, target in targets.items():
                achieved = achieved_concentrations.get(nutrient, 0.0)
                if target > 0:
                    deviation = ((achieved - target) / target) * 100.0
                else:
                    deviation = 0.0
                deviations_percent[nutrient] = deviation
            
            # Calculate ionic balance
            ionic_balance_error = self._calculate_ionic_balance_error(achieved_concentrations)
            
            # Clean up tiny dosages
            min_threshold = 0.001
            cleaned_dosages = {}
            for name, dosage in dosages.items():
                cleaned_dosages[name] = dosage if dosage >= min_threshold else 0.0
            
            active_fertilizers = len([d for d in cleaned_dosages.values() if d > 0])
            total_dosage = sum(cleaned_dosages.values())
            
            return LinearProgrammingResult(
                dosages_g_per_L=cleaned_dosages,
                achieved_concentrations=achieved_concentrations,
                deviations_percent=deviations_percent,
                optimization_status="Optimal",
                objective_value=lp.value(prob.objective),
                ionic_balance_error=ionic_balance_error,
                solver_time_seconds=0.0,  # Will be set by caller
                active_fertilizers=active_fertilizers,
                total_dosage=total_dosage
            )
        
        else:
            print(f"[LP] Optimization failed: {lp.LpStatus[prob.status]}")
            return self._create_fallback_result(fertilizers, targets, water)
    
    def _optimize_with_scipy(self, 
                            fertilizers: List,
                            targets: Dict[str, float],
                            water: Dict[str, float],
                            volume_liters: float) -> LinearProgrammingResult:
        """
        Optimize using SciPy optimization (fallback method)
        """
        print(f"[LP] Using SciPy solver for optimization...")
        
        n_fertilizers = len(fertilizers)
        n_nutrients = len(targets)
        
        # Decision variables: [dosage_1, dosage_2, ..., dosage_n]
        
        # Build coefficient matrix for nutrient contributions
        # A[i,j] = contribution of fertilizer j to nutrient i per g/L
        A_matrix = np.zeros((n_nutrients, n_fertilizers))
        nutrient_list = list(targets.keys())
        
        for i, nutrient in enumerate(nutrient_list):
            for j, fert in enumerate(fertilizers):
                cation_content = fert.composition.cations.get(nutrient, 0.0)
                anion_content = fert.composition.anions.get(nutrient, 0.0)
                total_content = cation_content + anion_content
                
                if total_content > 0:
                    # Convert to mg/L contribution per g/L dosage
                    contribution_factor = total_content * fert.chemistry.purity / 100.0 * 1000.0 / 100.0
                    A_matrix[i, j] = contribution_factor
        
        # Objective function: minimize sum of squared deviations
        def objective(x):
            dosages = x
            achieved = A_matrix @ dosages + np.array([water.get(nutrient, 0.0) for nutrient in nutrient_list])
            targets_array = np.array([targets[nutrient] for nutrient in nutrient_list])
            
            # Calculate weighted deviations
            deviations = np.abs(achieved - targets_array) / np.maximum(targets_array, 1e-6)
            
            # Primary objective: minimize deviations
            deviation_penalty = np.sum(deviations ** 2) * self.objective_weights['deviation_minimization']
            
            # Secondary objective: minimize total dosage
            dosage_penalty = np.sum(dosages) * self.objective_weights['dosage_minimization']
            
            return deviation_penalty + dosage_penalty
        
        # Constraints
        bounds = Bounds(0, self.max_individual_dosage)  # Individual dosage bounds
        
        # Total dosage constraint
        total_dosage_constraint = LinearConstraint(
            np.ones(n_fertilizers), 0, self.max_total_dosage
        )
        
        # Initial guess (small amounts of each fertilizer)
        x0 = np.full(n_fertilizers, 0.1)
        
        # Solve optimization
        try:
            result = minimize(
                objective, x0,
                method='trust-constr',
                bounds=bounds,
                constraints=[total_dosage_constraint],
                options={'maxiter': 1000, 'disp': False}
            )
            
            if result.success:
                print(f"[LP] SciPy optimization successful!")
                
                dosages = {}
                for i, fert in enumerate(fertilizers):
                    dosage_value = max(0.0, result.x[i])
                    dosages[fert.name] = dosage_value
                
                # Calculate achieved concentrations and deviations
                achieved_concentrations = self._calculate_achieved_concentrations(
                    dosages, water, fertilizers
                )
                
                deviations_percent = {}
                for nutrient, target in targets.items():
                    achieved = achieved_concentrations.get(nutrient, 0.0)
                    if target > 0:
                        deviation = ((achieved - target) / target) * 100.0
                    else:
                        deviation = 0.0
                    deviations_percent[nutrient] = deviation
                
                # Calculate ionic balance
                ionic_balance_error = self._calculate_ionic_balance_error(achieved_concentrations)
                
                # Clean up tiny dosages
                min_threshold = 0.001
                cleaned_dosages = {}
                for name, dosage in dosages.items():
                    cleaned_dosages[name] = dosage if dosage >= min_threshold else 0.0
                
                active_fertilizers = len([d for d in cleaned_dosages.values() if d > 0])
                total_dosage = sum(cleaned_dosages.values())
                
                return LinearProgrammingResult(
                    dosages_g_per_L=cleaned_dosages,
                    achieved_concentrations=achieved_concentrations,
                    deviations_percent=deviations_percent,
                    optimization_status="Optimal",
                    objective_value=result.fun,
                    ionic_balance_error=ionic_balance_error,
                    solver_time_seconds=0.0,  # Will be set by caller
                    active_fertilizers=active_fertilizers,
                    total_dosage=total_dosage
                )
            
            else:
                print(f"[LP] SciPy optimization failed: {result.message}")
                return self._create_fallback_result(fertilizers, targets, water)
                
        except Exception as e:
            print(f"[LP] SciPy optimization error: {e}")
            return self._create_fallback_result(fertilizers, targets, water)
    
    def _calculate_achieved_concentrations(self, 
                                         dosages: Dict[str, float],
                                         water: Dict[str, float],
                                         fertilizers: List) -> Dict[str, float]:
        """Calculate achieved nutrient concentrations from dosages"""
        achieved = water.copy()
        
        fert_map = {f.name: f for f in fertilizers}
        
        for fert_name, dosage_g_l in dosages.items():
            if dosage_g_l > 0 and fert_name in fert_map:
                fertilizer = fert_map[fert_name]
                dosage_mg_l = dosage_g_l * 1000
                
                # Add contributions from cations
                for element, content_percent in fertilizer.composition.cations.items():
                    if content_percent > 0:
                        contribution = self.nutrient_calc.calculate_element_contribution(
                            dosage_mg_l, content_percent, fertilizer.chemistry.purity
                        )
                        achieved[element] = achieved.get(element, 0) + contribution
                
                # Add contributions from anions
                for element, content_percent in fertilizer.composition.anions.items():
                    if content_percent > 0:
                        contribution = self.nutrient_calc.calculate_element_contribution(
                            dosage_mg_l, content_percent, fertilizer.chemistry.purity
                        )
                        achieved[element] = achieved.get(element, 0) + contribution
        
        return achieved
    
    def _calculate_ionic_balance_error(self, concentrations: Dict[str, float]) -> float:
        """Calculate ionic balance error in %"""
        cation_elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'Fe', 'Mn', 'Zn', 'Cu']
        anion_elements = ['NO3', 'H2PO4', 'SO4', 'Cl', 'HCO3']
        
        try:
            cation_meq = sum(
                self.nutrient_calc.convert_mg_to_meq_direct(concentrations.get(elem, 0), elem)
                for elem in cation_elements
            )
            anion_meq = sum(
                self.nutrient_calc.convert_mg_to_meq_direct(concentrations.get(elem, 0), elem)
                for elem in anion_elements
            )
            
            if cation_meq > 0 or anion_meq > 0:
                balance_error = abs(cation_meq - anion_meq) / max(cation_meq, anion_meq, 1.0) * 100
            else:
                balance_error = 0.0
                
            return balance_error
            
        except Exception:
            return 0.0
    
    def _create_fallback_result(self, 
                               fertilizers: List,
                               targets: Dict[str, float],
                               water: Dict[str, float]) -> LinearProgrammingResult:
        """Create fallback result when optimization fails"""
        print(f"[LP] Creating fallback solution...")
        
        # Simple fallback: distribute dosages evenly among useful fertilizers
        dosages = {}
        total_budget = 3.0  # g/L total
        
        useful_fertilizers = []
        for fert in fertilizers:
            total_content = (sum(fert.composition.cations.values()) + 
                           sum(fert.composition.anions.values()))
            if total_content > 5:  # Has significant nutrient content
                useful_fertilizers.append(fert)
        
        if useful_fertilizers:
            dosage_per_fert = total_budget / len(useful_fertilizers)
            for fert in useful_fertilizers:
                dosages[fert.name] = min(dosage_per_fert, 1.0)
        
        # Fill with zeros for unused fertilizers
        for fert in fertilizers:
            if fert.name not in dosages:
                dosages[fert.name] = 0.0
        
        achieved_concentrations = self._calculate_achieved_concentrations(dosages, water, fertilizers)
        
        deviations_percent = {}
        for nutrient, target in targets.items():
            achieved = achieved_concentrations.get(nutrient, 0.0)
            if target > 0:
                deviation = ((achieved - target) / target) * 100.0
            else:
                deviation = 0.0
            deviations_percent[nutrient] = deviation
        
        return LinearProgrammingResult(
            dosages_g_per_L=dosages,
            achieved_concentrations=achieved_concentrations,
            deviations_percent=deviations_percent,
            optimization_status="Fallback",
            objective_value=999999.0,
            ionic_balance_error=self._calculate_ionic_balance_error(achieved_concentrations),
            solver_time_seconds=0.0,
            active_fertilizers=len([d for d in dosages.values() if d > 0]),
            total_dosage=sum(dosages.values())
        )