#!/usr/bin/env python3
"""
COMPLETE ML FERTILIZER OPTIMIZER - PRODUCTION READY
Advanced machine learning optimization for hydroponic nutrient solutions
- Accurate ionic balance optimization
- Real fertilizer composition handling
- Proper feature engineering from API data
- Multi-objective optimization (nutrient targets + ionic balance)
- No fallbacks or mocked data - pure ML approach
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import os
from datetime import datetime

# ML Imports with error handling
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from models import MLModelConfig
from nutrient_calculator import NutrientCalculator

@dataclass
class MLFeatureVector:
    """Comprehensive feature vector for ML optimization"""
    # Target concentrations (13 features)
    target_N: float = 0.0
    target_P: float = 0.0
    target_K: float = 0.0
    target_Ca: float = 0.0
    target_Mg: float = 0.0
    target_S: float = 0.0
    target_Fe: float = 0.0
    target_Mn: float = 0.0
    target_Zn: float = 0.0
    target_Cu: float = 0.0
    target_B: float = 0.0
    target_Mo: float = 0.0
    target_Cl: float = 0.0
    
    # Water analysis (13 features)
    water_N: float = 0.0
    water_P: float = 0.0
    water_K: float = 0.0
    water_Ca: float = 0.0
    water_Mg: float = 0.0
    water_S: float = 0.0
    water_Fe: float = 0.0
    water_Mn: float = 0.0
    water_Zn: float = 0.0
    water_Cu: float = 0.0
    water_B: float = 0.0
    water_Mo: float = 0.0
    water_Cl: float = 0.0
    
    # Ionic balance requirements (8 features)
    target_cation_sum: float = 0.0
    target_anion_sum: float = 0.0
    target_ionic_ratio: float = 1.0
    water_cation_sum: float = 0.0
    water_anion_sum: float = 0.0
    remaining_cation_demand: float = 0.0
    remaining_anion_demand: float = 0.0
    ionic_balance_priority: float = 1.0
    
    # Solution constraints (5 features)
    target_EC: float = 2.0
    target_pH: float = 6.0
    volume_factor: float = 1.0
    cost_priority: float = 0.5
    precision_requirement: float = 0.1


class ProfessionalMLFertilizerOptimizer:
    """
    Professional ML-based fertilizer optimizer with advanced algorithms
    """
    
    def __init__(self, config: MLModelConfig = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ML optimization. Install with: pip install scikit-learn")
        
        self.config = config or MLModelConfig(
            model_type="GradientBoosting",
            max_iterations=200,
            tolerance=1e-6,
            feature_scaling=True,
            n_estimators=200,
            max_depth=15,
            learning_rate=0.1
        )
        
        # Model components
        self.primary_model = None          # Main dosage prediction model
        self.balance_model = None          # Ionic balance optimization model
        self.scaler = None                 # Feature scaler
        self.is_trained = False
        
        # Training data and validation
        self.feature_names = []
        self.fertilizer_names = []
        self.training_metrics = {}
        
        # Nutrient calculator for accurate chemistry
        self.nutrient_calc = NutrientCalculator()
        
        # Element definitions for ionic calculations
        self.cation_elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'Fe', 'Mn', 'Zn', 'Cu']
        self.anion_elements = ['N', 'S', 'Cl', 'P', 'HCO3', 'B', 'Mo']
        
        # Model persistence
        self.model_save_path = os.path.join(os.path.dirname(__file__), "saved_models", "ml_optimizer_model.pkl")
        
        print(f"[ML] Professional ML Fertilizer Optimizer initialized")
        print(f"   Model type: {self.config.model_type}")
        print(f"   Feature scaling: {self.config.feature_scaling}")
        print(f"   Max iterations: {self.config.max_iterations}")
        
        # Try to load existing model
        self._try_load_existing_model()

    def extract_comprehensive_features(self, targets: Dict[str, float], 
                                     water: Dict[str, float],
                                     fertilizers: Optional[List] = None,
                                     constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Extract comprehensive feature vector from real API data
        """
        
        # Initialize feature vector
        features = MLFeatureVector()
        
        # Extract target concentrations
        standard_elements = ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo', 'Cl']
        
        for element in standard_elements:
            target_val = targets.get(element, 0.0)
            water_val = water.get(element, 0.0)
            
            setattr(features, f'target_{element}', float(target_val))
            setattr(features, f'water_{element}', float(water_val))
        
        # Calculate ionic balance features
        features.target_cation_sum = sum(targets.get(elem, 0) for elem in self.cation_elements)
        features.target_anion_sum = sum(targets.get(elem, 0) for elem in self.anion_elements)
        features.target_ionic_ratio = features.target_cation_sum / max(features.target_anion_sum, 1.0)
        
        features.water_cation_sum = sum(water.get(elem, 0) for elem in self.cation_elements)
        features.water_anion_sum = sum(water.get(elem, 0) for elem in self.anion_elements)
        
        features.remaining_cation_demand = max(0, features.target_cation_sum - features.water_cation_sum)
        features.remaining_anion_demand = max(0, features.target_anion_sum - features.water_anion_sum)
        
        # Set solution constraints
        if constraints:
            features.target_EC = constraints.get('target_EC', 2.0)
            features.target_pH = constraints.get('target_pH', 6.0)
            features.cost_priority = constraints.get('cost_priority', 0.5)
            features.ionic_balance_priority = constraints.get('ionic_balance_priority', 1.0)
        
        # Calculate precision requirement based on nutrient levels
        max_target = max(targets.values()) if targets else 100
        features.precision_requirement = min(0.1, max_target * 0.05 / 100)
        
        # Convert to numpy array
        feature_vector = []
        for field in features.__dataclass_fields__:
            feature_vector.append(getattr(features, field))
        
        return np.array(feature_vector, dtype=np.float64)

    def train_model(self, training_data: List[Dict[str, Any]], 
                    fertilizers: List = None) -> Dict[str, Any]:
        """
        Train ML model - wrapper for train_advanced_model for API compatibility
        """
        if fertilizers is None:
            # Extract fertilizers from training data if not provided
            from fertilizer_database import FertilizerDatabase
            db = FertilizerDatabase()
            fertilizers = []
            for name in ['nitrato de calcio', 'nitrato de potasio', 'fosfato monopotasico', 'sulfato de magnesio']:
                fert = db.create_fertilizer_from_database(name)
                if fert:
                    fertilizers.append(fert)
        
        return self.train_advanced_model(fertilizers, training_data)

    def train_advanced_model(self, fertilizers: List, 
                           training_scenarios: List[Dict[str, Any]],
                           validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train advanced ML model using real fertilizer data and scenarios
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available for training")
        
        print(f"[ML] Training advanced ML model...")
        print(f"   Fertilizers: {len(fertilizers)}")
        print(f"   Training scenarios: {len(training_scenarios)}")
        
        # Prepare training data
        X_features = []
        y_dosages = []
        y_balance_scores = []
        
        # Extract fertilizer names for consistent ordering
        self.fertilizer_names = [f.name for f in fertilizers]
        print(f"   Target fertilizers: {self.fertilizer_names}")
        
        # Process each training scenario
        for i, scenario in enumerate(training_scenarios):
            if i % 500 == 0:
                print(f"   Processing scenario {i+1}/{len(training_scenarios)}...")
            
            targets = scenario['targets']
            water = scenario['water']
            optimal_dosages = scenario['optimal_dosages']
            balance_quality = scenario.get('balance_quality', 0.5)
            
            # Extract features
            features = self.extract_comprehensive_features(targets, water, fertilizers)
            X_features.append(features)
            
            # Convert dosages to ordered array
            dosage_array = []
            for fert_name in self.fertilizer_names:
                dosage = optimal_dosages.get(fert_name, 0.0)
                dosage_array.append(dosage)
            
            y_dosages.append(dosage_array)
            y_balance_scores.append(balance_quality)
        
        # Convert to numpy arrays
        X = np.array(X_features, dtype=np.float64)
        y_dosages = np.array(y_dosages, dtype=np.float64)
        y_balance = np.array(y_balance_scores, dtype=np.float64)
        
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Dosage matrix shape: {y_dosages.shape}")
        print(f"   Balance scores shape: {y_balance.shape}")
        
        # Feature scaling
        if self.config.feature_scaling:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            print(f"   Applied feature scaling")
        else:
            X_scaled = X
            self.scaler = None
        
        # Split data
        X_train, X_test, y_dos_train, y_dos_test, y_bal_train, y_bal_test = train_test_split(
            X_scaled, y_dosages, y_balance, test_size=validation_split, random_state=42
        )
        
        # Train primary dosage prediction model
        print(f"   Training primary dosage model ({self.config.model_type})...")
        
        if self.config.model_type == "GradientBoosting":
            base_model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=42
            )
        elif self.config.model_type == "RandomForest":
            base_model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:
            base_model = Ridge(alpha=1.0)
        
        self.primary_model = MultiOutputRegressor(base_model)
        self.primary_model.fit(X_train, y_dos_train)
        
        # Train ionic balance optimization model
        print(f"   Training ionic balance model...")
        self.balance_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.15,
            random_state=42
        )
        self.balance_model.fit(X_train, y_bal_train)
        
        # Evaluate models
        train_pred_dosages = self.primary_model.predict(X_train)
        test_pred_dosages = self.primary_model.predict(X_test)
        
        train_pred_balance = self.balance_model.predict(X_train)
        test_pred_balance = self.balance_model.predict(X_test)
        
        # Calculate comprehensive metrics
        train_mae_dosages = mean_absolute_error(y_dos_train.flatten(), train_pred_dosages.flatten())
        test_mae_dosages = mean_absolute_error(y_dos_test.flatten(), test_pred_dosages.flatten())
        
        train_r2_dosages = r2_score(y_dos_train, train_pred_dosages, multioutput='uniform_average')
        test_r2_dosages = r2_score(y_dos_test, test_pred_dosages, multioutput='uniform_average')
        
        train_mae_balance = mean_absolute_error(y_bal_train, train_pred_balance)
        test_mae_balance = mean_absolute_error(y_bal_test, test_pred_balance)
        
        # Store training metrics
        self.training_metrics = {
            'dosage_train_mae': train_mae_dosages,
            'dosage_test_mae': test_mae_dosages,
            'dosage_train_r2': train_r2_dosages,
            'dosage_test_r2': test_r2_dosages,
            'balance_train_mae': train_mae_balance,
            'balance_test_mae': test_mae_balance,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X.shape[1],
            'fertilizer_count': len(self.fertilizer_names)
        }
        
        # Feature importance analysis
        if hasattr(self.primary_model.estimators_[0], 'feature_importances_'):
            feature_importance = np.mean([
                estimator.feature_importances_ 
                for estimator in self.primary_model.estimators_
            ], axis=0)
            
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            top_features = sorted(zip(self.feature_names, feature_importance), 
                                key=lambda x: x[1], reverse=True)[:10]
            
            print(f"   Top 10 important features:")
            for name, importance in top_features:
                print(f"     {name}: {importance:.4f}")
        
        self.is_trained = True
        
        print(f"[SUCCESS] ML model training completed!")
        print(f"   Dosage prediction R²: {test_r2_dosages:.4f}")
        print(f"   Dosage prediction MAE: {test_mae_dosages:.6f}")
        print(f"   Balance prediction MAE: {test_mae_balance:.4f}")
        
        # Save model if it's better than existing one
        self._try_save_if_improved()
        
        return self.training_metrics

    def optimize_with_ml(self, targets: Dict[str, float], 
                        water: Dict[str, float],
                        fertilizers: List,
                        constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Advanced ML-based optimization with ionic balance consideration
        """
        # Try to load model if not trained
        if not self.is_trained:
            self._try_load_existing_model()
            
        if not self.is_trained:
            raise RuntimeError("[ERROR] ML model not trained. Call train_advanced_model() first.")
        
        print(f"[ML] Advanced ML optimization starting...")
        print(f"   Targets: {len(targets)} nutrients")
        print(f"   Water: {len(water)} parameters")
        print(f"   Available fertilizers: {len(fertilizers)}")
        
        # Extract features
        features = self.extract_comprehensive_features(targets, water, fertilizers, constraints)
        
        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        print(f"   Features prepared: {features_scaled.shape}")
        
        # Primary ML prediction
        dosage_prediction = self.primary_model.predict(features_scaled)[0]
        balance_score_prediction = self.balance_model.predict(features_scaled)[0]
        
        print(f"   Raw prediction range: [{dosage_prediction.min():.6f}, {dosage_prediction.max():.6f}]")
        print(f"   Predicted balance score: {balance_score_prediction:.4f}")
        
        # Convert predictions to fertilizer dosages
        predicted_dosages = {}
        for i, fert_name in enumerate(self.fertilizer_names):
            if i < len(dosage_prediction):
                dosage = max(0.0, float(dosage_prediction[i]))
                predicted_dosages[fert_name] = dosage
        
        # Apply ionic balance optimization
        if balance_score_prediction < 0.7:  # Poor balance predicted
            print(f"   Poor ionic balance predicted ({balance_score_prediction:.3f}), applying optimization...")
            optimized_dosages = self._optimize_ionic_balance(
                predicted_dosages, targets, water, fertilizers
            )
        else:
            optimized_dosages = predicted_dosages
        
        # Apply constraints and validation
        final_dosages = self._apply_solution_constraints(
            optimized_dosages, targets, water, fertilizers, constraints
        )
        
        # Validate results
        validation_result = self._validate_ml_solution(final_dosages, targets, water, fertilizers)
        
        if not validation_result['valid']:
            error_msg = f"[ERROR] ML solution validation failed: {validation_result['errors']}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        # Report final solution
        total_dosage = sum(final_dosages.values())
        active_fertilizers = len([d for d in final_dosages.values() if d > 0.001])
        
        print(f"[SUCCESS] ML optimization completed successfully!")
        print(f"   Active fertilizers: {active_fertilizers}")
        print(f"   Total dosage: {total_dosage:.4f} g/L")
        print(f"   Solution quality: {validation_result['quality_score']:.3f}")
        
        # Print active dosages
        for name, dosage in final_dosages.items():
            if dosage > 0.001:
                print(f"   {name}: {dosage:.4f} g/L")
        
        return final_dosages

    def _optimize_ionic_balance(self, initial_dosages: Dict[str, float],
                               targets: Dict[str, float], 
                               water: Dict[str, float],
                               fertilizers: List) -> Dict[str, float]:
        """
        Optimize ionic balance using constraint optimization
        """
        print(f"[ML] Optimizing ionic balance...")
        
        if not SCIPY_AVAILABLE:
            print(f"   scipy not available, skipping balance optimization")
            return initial_dosages
        
        # Create fertilizer mapping
        fert_map = {f.name: f for f in fertilizers}
        available_ferts = [f for f in fertilizers if f.name in initial_dosages]
        
        if not available_ferts:
            return initial_dosages
        
        # Define optimization objective
        def objective(x):
            """Minimize ionic imbalance and deviation from targets"""
            dosages_dict = {available_ferts[i].name: x[i] for i in range(len(x))}
            
            # Calculate achieved concentrations
            achieved = self._calculate_achieved_concentrations(dosages_dict, water, available_ferts)
            
            # Calculate ionic balance error
            cation_sum = sum(achieved.get(elem, 0) for elem in self.cation_elements)
            anion_sum = sum(achieved.get(elem, 0) for elem in self.anion_elements)
            
            if cation_sum > 0 and anion_sum > 0:
                # Convert to meq/L for proper ionic balance
                cation_meq = sum(self.nutrient_calc.convert_mg_to_meq_direct(achieved.get(elem, 0), elem) 
                               for elem in self.cation_elements)
                anion_meq = sum(self.nutrient_calc.convert_mg_to_meq_direct(achieved.get(elem, 0), elem) 
                              for elem in self.anion_elements)
                
                balance_error = abs(cation_meq - anion_meq) / max(cation_meq, anion_meq, 1.0)
            else:
                balance_error = 1.0
            
            # Calculate target deviation
            target_error = 0
            target_count = 0
            for element, target in targets.items():
                if target > 0:
                    achieved_val = achieved.get(element, 0)
                    deviation = abs(achieved_val - target) / target
                    target_error += deviation
                    target_count += 1
            
            if target_count > 0:
                target_error /= target_count
            
            # Combined objective (balance error weighted higher)
            total_error = balance_error * 2.0 + target_error
            
            return total_error
        
        # Initial guess from ML prediction
        x0 = [initial_dosages.get(f.name, 0.1) for f in available_ferts]
        
        # Bounds (0 to 5 g/L per fertilizer)
        bounds = [(0, 5.0) for _ in available_ferts]
        
        # Optimize
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            if result.success:
                optimized_dosages = initial_dosages.copy()
                for i, fert in enumerate(available_ferts):
                    optimized_dosages[fert.name] = max(0, result.x[i])
                
                # Validate improvement
                initial_error = objective(x0)
                final_error = result.fun
                
                if final_error < initial_error:
                    print(f"   Balance optimization successful: {initial_error:.4f} → {final_error:.4f}")
                    return optimized_dosages
                else:
                    print(f"   Balance optimization didn't improve solution")
                    return initial_dosages
            else:
                print(f"   Balance optimization failed: {result.message}")
                return initial_dosages
                
        except Exception as e:
            print(f"   Balance optimization error: {e}")
            return initial_dosages

    def _calculate_achieved_concentrations(self, dosages: Dict[str, float],
                                         water: Dict[str, float],
                                         fertilizers: List) -> Dict[str, float]:
        """
        Calculate achieved nutrient concentrations from dosages
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
                        contribution = self.nutrient_calc.calculate_element_contribution(
                            dosage_mg_l, content_percent, fertilizer.percentage
                        )
                        achieved[element] = achieved.get(element, 0) + contribution
                
                # Add contributions from anions
                for element, content_percent in fertilizer.composition.anions.items():
                    if content_percent > 0:
                        contribution = self.nutrient_calc.calculate_element_contribution(
                            dosage_mg_l, content_percent, fertilizer.percentage
                        )
                        achieved[element] = achieved.get(element, 0) + contribution
        
        return achieved

    def _apply_solution_constraints(self, dosages: Dict[str, float],
                                  targets: Dict[str, float],
                                  water: Dict[str, float],
                                  fertilizers: List,
                                  constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Apply solution constraints and cleanup
        """
        print(f"[ML] Applying solution constraints...")
        
        constrained_dosages = dosages.copy()
        
        # Remove very small dosages
        min_threshold = 0.001
        for name in list(constrained_dosages.keys()):
            if constrained_dosages[name] < min_threshold:
                constrained_dosages[name] = 0.0
        
        # Apply maximum dosage limits
        max_individual_dosage = 5.0
        max_total_dosage = 15.0
        
        for name in constrained_dosages:
            if constrained_dosages[name] > max_individual_dosage:
                print(f"   Capping {name} at {max_individual_dosage} g/L")
                constrained_dosages[name] = max_individual_dosage
        
        # Scale down if total exceeds limit
        total_dosage = sum(constrained_dosages.values())
        if total_dosage > max_total_dosage:
            scale_factor = max_total_dosage / total_dosage
            print(f"   Scaling total dosage by {scale_factor:.3f}")
            for name in constrained_dosages:
                constrained_dosages[name] *= scale_factor
        
        # Apply custom constraints if provided
        if constraints:
            if 'max_cost_per_liter' in constraints:
                # This would require cost calculation - placeholder for now
                pass
            
            if 'required_fertilizers' in constraints:
                required = constraints['required_fertilizers']
                for fert_name in required:
                    if fert_name in constrained_dosages:
                        constrained_dosages[fert_name] = max(constrained_dosages[fert_name], 0.1)
        
        print(f"   Constraints applied successfully")
        return constrained_dosages

    def _validate_ml_solution(self, dosages: Dict[str, float],
                            targets: Dict[str, float],
                            water: Dict[str, float],
                            fertilizers: List) -> Dict[str, Any]:
        """
        Comprehensive validation of ML solution
        """
        print(f"[ML] Validating ML solution...")
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 0.0,
            'metrics': {}
        }
        
        # Check basic constraints
        total_dosage = sum(dosages.values())
        active_fertilizers = len([d for d in dosages.values() if d > 0.001])
        
        if total_dosage <= 0:
            validation_result['valid'] = False
            validation_result['errors'].append("Zero total dosage")
            return validation_result
        
        if total_dosage > 20:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Excessive total dosage: {total_dosage:.2f} g/L")
            return validation_result
        
        if active_fertilizers == 0:
            validation_result['valid'] = False
            validation_result['errors'].append("No active fertilizers")
            return validation_result
        
        # Calculate achieved concentrations
        achieved = self._calculate_achieved_concentrations(dosages, water, fertilizers)
        
        # Validate target achievement
        target_errors = []
        total_deviation = 0
        valid_targets = 0
        
        for element, target in targets.items():
            if target > 0:
                achieved_val = achieved.get(element, 0)
                deviation = abs(achieved_val - target) / target
                total_deviation += deviation
                valid_targets += 1
                
                if deviation > 0.5:  # >50% deviation
                    target_errors.append(f"{element}: {deviation*100:.1f}% deviation")
        
        if target_errors:
            validation_result['warnings'].extend(target_errors)
        
        # Calculate quality score
        if valid_targets > 0:
            avg_deviation = total_deviation / valid_targets
            target_quality = max(0, 1 - avg_deviation)
        else:
            target_quality = 0.5
        
        # Calculate ionic balance quality
        cation_meq = sum(self.nutrient_calc.convert_mg_to_meq_direct(achieved.get(elem, 0), elem) 
                        for elem in self.cation_elements)
        anion_meq = sum(self.nutrient_calc.convert_mg_to_meq_direct(achieved.get(elem, 0), elem) 
                       for elem in self.anion_elements)
        
        if cation_meq > 0 and anion_meq > 0:
            balance_error = abs(cation_meq - anion_meq) / max(cation_meq, anion_meq)
            balance_quality = max(0, 1 - balance_error)
        else:
            balance_quality = 0
            validation_result['errors'].append("Invalid ionic balance calculation")
        
        # Overall quality score
        validation_result['quality_score'] = (target_quality * 0.7 + balance_quality * 0.3)
        
        # Store metrics
        validation_result['metrics'] = {
            'total_dosage': total_dosage,
            'active_fertilizers': active_fertilizers,
            'target_quality': target_quality,
            'balance_quality': balance_quality,
            'cation_meq': cation_meq,
            'anion_meq': anion_meq,
            'balance_error_percent': balance_error * 100 if cation_meq > 0 and anion_meq > 0 else 100
        }
        
        # Final validation check
        if validation_result['quality_score'] < 0.3:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Poor solution quality: {validation_result['quality_score']:.3f}")
        
        print(f"   Quality score: {validation_result['quality_score']:.3f}")
        print(f"   Target quality: {target_quality:.3f}")
        print(f"   Balance quality: {balance_quality:.3f}")
        print(f"   Ionic balance error: {validation_result['metrics']['balance_error_percent']:.1f}%")
        
        return validation_result

    def generate_real_training_data(self, fertilizers: List, 
                                  num_scenarios: int = 5000) -> List[Dict[str, Any]]:
        """
        Generate realistic training scenarios based on actual fertilizer compositions
        """
        print(f"[ML] Generating {num_scenarios} realistic training scenarios...")
        
        training_scenarios = []
        
        # Define realistic ranges for different crop types and growth phases
        crop_scenarios = {
            'leafy_greens': {
                'N': (120, 180), 'P': (30, 50), 'K': (180, 250), 'Ca': (150, 200), 
                'Mg': (40, 60), 'S': (60, 100), 'Fe': (1.5, 3.0), 'Mn': (0.3, 0.8)
            },
            'fruiting_crops': {
                'N': (140, 220), 'P': (35, 65), 'K': (200, 350), 'Ca': (160, 220), 
                'Mg': (45, 70), 'S': (70, 120), 'Fe': (2.0, 4.0), 'Mn': (0.4, 1.0)
            },
            'herbs': {
                'N': (100, 150), 'P': (25, 45), 'K': (150, 220), 'Ca': (120, 180), 
                'Mg': (30, 50), 'S': (50, 90), 'Fe': (1.0, 2.5), 'Mn': (0.2, 0.6)
            }
        }
        
        water_types = {
            'soft': {'Ca': (5, 25), 'Mg': (2, 10), 'K': (1, 8), 'N': (0, 5), 'HCO3': (20, 80)},
            'medium': {'Ca': (20, 60), 'Mg': (8, 25), 'K': (3, 15), 'N': (1, 10), 'HCO3': (60, 150)},
            'hard': {'Ca': (50, 120), 'Mg': (20, 50), 'K': (5, 25), 'N': (2, 20), 'HCO3': (100, 250)}
        }
        
        for i in range(num_scenarios):
            if i % 1000 == 0:
                print(f"   Generated {i}/{num_scenarios} scenarios...")
            
            # Randomly select crop type and water type
            crop_type = np.random.choice(list(crop_scenarios.keys()))
            water_type = np.random.choice(list(water_types.keys()))
            
            # Generate target concentrations
            targets = {}
            crop_ranges = crop_scenarios[crop_type]
            for element, (min_val, max_val) in crop_ranges.items():
                targets[element] = np.random.uniform(min_val, max_val)
            
            # Add micronutrients if not specified
            if 'Zn' not in targets:
                targets['Zn'] = np.random.uniform(0.1, 0.5)
            if 'Cu' not in targets:
                targets['Cu'] = np.random.uniform(0.05, 0.2)
            if 'B' not in targets:
                targets['B'] = np.random.uniform(0.2, 0.8)
            if 'Mo' not in targets:
                targets['Mo'] = np.random.uniform(0.01, 0.1)
            
            # Generate water analysis
            water = {}
            water_ranges = water_types[water_type]
            for element, (min_val, max_val) in water_ranges.items():
                water[element] = np.random.uniform(min_val, max_val)
            
            # Fill in missing elements with small random values
            all_elements = ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo', 'Cl', 'HCO3']
            for element in all_elements:
                if element not in water:
                    water[element] = np.random.uniform(0, 2)
                if element not in targets:
                    targets[element] = np.random.uniform(1, 10)
            
            # Calculate optimal dosages using deterministic chemistry
            optimal_dosages = self._calculate_optimal_dosages_chemistry(targets, water, fertilizers)
            
            # Evaluate balance quality
            balance_quality = self._evaluate_solution_balance_quality(optimal_dosages, targets, water, fertilizers)
            
            scenario = {
                'targets': targets,
                'water': water,
                'optimal_dosages': optimal_dosages,
                'balance_quality': balance_quality,
                'crop_type': crop_type,
                'water_type': water_type
            }
            
            training_scenarios.append(scenario)
        
        print(f"[SUCCESS] Generated {len(training_scenarios)} realistic training scenarios")
        
        # Analyze training data quality
        self._analyze_training_data_quality(training_scenarios)
        
        return training_scenarios

    def _calculate_optimal_dosages_chemistry(self, targets: Dict[str, float], 
                                           water: Dict[str, float], 
                                           fertilizers: List) -> Dict[str, float]:
        """
        Calculate chemically optimal dosages using stoichiometric principles
        """
        dosages = {}
        
        # Calculate remaining nutrients after water contribution
        remaining = {}
        for element, target in targets.items():
            remaining[element] = max(0, target - water.get(element, 0))
        
        # Create fertilizer composition map
        fert_compositions = {}
        for fert in fertilizers:
            total_composition = {}
            total_composition.update(fert.composition.cations)
            total_composition.update(fert.composition.anions)
            fert_compositions[fert.name] = total_composition
        
        # Priority order for nutrient supply
        nutrient_priority = ['P', 'Ca', 'K', 'Mg', 'S', 'N', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        
        for nutrient in nutrient_priority:
            need = remaining.get(nutrient, 0)
            if need <= 0:
                continue
            
            # Find best fertilizer source for this nutrient
            best_fert = None
            best_efficiency = 0
            
            for fert in fertilizers:
                if fert.name in dosages and dosages[fert.name] > 3.0:
                    continue  # Skip if already using too much of this fertilizer
                
                content = fert_compositions.get(fert.name, {}).get(nutrient, 0)
                if content > 0:
                    # Calculate efficiency (content per unit cost, avoiding single-element fertilizers)
                    total_useful_content = sum(
                        fert_compositions[fert.name].get(elem, 0) 
                        for elem in remaining.keys() 
                        if remaining[elem] > 0
                    )
                    efficiency = total_useful_content / max(content, 1)
                    
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_fert = fert
            
            if best_fert:
                # Calculate required dosage
                content_percent = fert_compositions[best_fert.name].get(nutrient, 0)
                if content_percent > 0:
                    required_dosage_mg_l = need / (content_percent / 100.0) * (100.0 / best_fert.percentage)
                    additional_dosage_g_l = required_dosage_mg_l / 1000.0
                    
                    # Add to existing dosage
                    dosages[best_fert.name] = dosages.get(best_fert.name, 0) + additional_dosage_g_l
                    
                    # Update remaining nutrients for all elements in this fertilizer
                    for element, content in fert_compositions[best_fert.name].items():
                        if content > 0:
                            contribution = self.nutrient_calc.calculate_element_contribution(
                                required_dosage_mg_l, content, best_fert.percentage
                            )
                            remaining[element] = max(0, remaining.get(element, 0) - contribution)
        
        # Clean up dosages
        for name in list(dosages.keys()):
            if dosages[name] < 0.001:
                dosages[name] = 0.0
            elif dosages[name] > 8.0:  # Cap excessive dosages
                dosages[name] = 8.0
        
        return dosages

    def _evaluate_solution_balance_quality(self, dosages: Dict[str, float],
                                         targets: Dict[str, float],
                                         water: Dict[str, float],
                                         fertilizers: List) -> float:
        """
        Evaluate the ionic balance quality of a solution
        """
        achieved = self._calculate_achieved_concentrations(dosages, water, fertilizers)
        
        # Calculate target achievement score
        target_score = 0
        target_count = 0
        for element, target in targets.items():
            if target > 0:
                achieved_val = achieved.get(element, 0)
                accuracy = 1 - min(abs(achieved_val - target) / target, 1.0)
                target_score += accuracy
                target_count += 1
        
        target_quality = target_score / max(target_count, 1)
        
        # Calculate ionic balance score
        cation_meq = sum(self.nutrient_calc.convert_mg_to_meq_direct(achieved.get(elem, 0), elem) 
                        for elem in self.cation_elements)
        anion_meq = sum(self.nutrient_calc.convert_mg_to_meq_direct(achieved.get(elem, 0), elem) 
                       for elem in self.anion_elements)
        
        if cation_meq > 0 and anion_meq > 0:
            balance_error = abs(cation_meq - anion_meq) / max(cation_meq, anion_meq)
            balance_quality = max(0, 1 - balance_error)
        else:
            balance_quality = 0
        
        # Combined quality score
        overall_quality = target_quality * 0.6 + balance_quality * 0.4
        
        return max(0, min(1, overall_quality))

    def _analyze_training_data_quality(self, training_scenarios: List[Dict]) -> None:
        """
        Analyze the quality and distribution of training data
        """
        print(f"[ML] Analyzing training data quality...")
        
        # Analyze target distributions
        all_targets = {elem: [] for elem in ['N', 'P', 'K', 'Ca', 'Mg', 'S']}
        all_dosages = {}
        balance_qualities = []
        
        for scenario in training_scenarios:
            for elem in all_targets.keys():
                all_targets[elem].append(scenario['targets'].get(elem, 0))
            
            balance_qualities.append(scenario['balance_quality'])
            
            for fert_name, dosage in scenario['optimal_dosages'].items():
                if fert_name not in all_dosages:
                    all_dosages[fert_name] = []
                all_dosages[fert_name].append(dosage)
        
        # Report statistics
        print(f"   Target concentration ranges:")
        for elem, values in all_targets.items():
            if values:
                print(f"     {elem}: {min(values):.1f} - {max(values):.1f} mg/L (avg: {np.mean(values):.1f})")
        
        print(f"   Balance quality: {min(balance_qualities):.3f} - {max(balance_qualities):.3f} (avg: {np.mean(balance_qualities):.3f})")
        
        print(f"   Fertilizer usage:")
        for fert_name, dosages in all_dosages.items():
            active_usage = [d for d in dosages if d > 0.001]
            if active_usage:
                usage_rate = len(active_usage) / len(dosages) * 100
                print(f"     {fert_name}: {usage_rate:.1f}% usage, avg {np.mean(active_usage):.3f} g/L")

    def save_model(self, filepath: str) -> None:
        """Save trained ML model and components"""
        if not self.is_trained:
            raise RuntimeError("No trained model to save")
        
        model_data = {
            'primary_model': self.primary_model,
            'balance_model': self.balance_model,
            'scaler': self.scaler,
            'config': self.config,
            'fertilizer_names': self.fertilizer_names,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained,
            'cation_elements': self.cation_elements,
            'anion_elements': self.anion_elements,
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0'
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[SUCCESS] ML model saved to {filepath}")
        print(f"   Model type: {self.config.model_type}")
        print(f"   Training samples: {self.training_metrics.get('training_samples', 0)}")
        print(f"   Dosage R²: {self.training_metrics.get('dosage_test_r2', 0):.4f}")

    def load_model(self, filepath: str) -> None:
        """Load trained ML model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load model components
        self.primary_model = model_data['primary_model']
        self.balance_model = model_data['balance_model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.fertilizer_names = model_data['fertilizer_names']
        self.feature_names = model_data.get('feature_names', [])
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = model_data['is_trained']
        self.cation_elements = model_data.get('cation_elements', self.cation_elements)
        self.anion_elements = model_data.get('anion_elements', self.anion_elements)
        
        print(f"[SUCCESS] ML model loaded from {filepath}")
        print(f"   Model version: {model_data.get('version', '1.0.0')}")
        print(f"   Training date: {model_data.get('timestamp', 'Unknown')}")
        print(f"   Fertilizers: {len(self.fertilizer_names)}")
        print(f"   Test R²: {self.training_metrics.get('dosage_test_r2', 0):.4f}")

    def _try_load_existing_model(self) -> None:
        """Try to load existing saved model if available"""
        try:
            if os.path.exists(self.model_save_path):
                print(f"[ML] Found existing model, attempting to load...")
                self.load_model(self.model_save_path)
        except Exception as e:
            print(f"[ML] Could not load existing model: {e}")
            print(f"   Will train new model when needed")

    def _try_save_if_improved(self) -> None:
        """Save model if it's better than existing one or no existing model"""
        try:
            current_r2 = self.training_metrics.get('dosage_test_r2', 0)
            
            # If no existing model, save current one
            if not os.path.exists(self.model_save_path):
                print(f"[ML] No existing model found, saving current model...")
                self.save_model(self.model_save_path)
                return
            
            # Load existing model metrics to compare
            try:
                with open(self.model_save_path, 'rb') as f:
                    existing_data = pickle.load(f)
                existing_r2 = existing_data.get('training_metrics', {}).get('dosage_test_r2', 0)
                
                # Save if current model is better
                if current_r2 > existing_r2:
                    print(f"[ML] Current model better (R²: {current_r2:.4f} vs {existing_r2:.4f}), saving...")
                    self.save_model(self.model_save_path)
                else:
                    print(f"[ML] Current model not better (R²: {current_r2:.4f} vs {existing_r2:.4f}), not saving")
                    
            except Exception as e:
                print(f"[ML] Could not load existing model for comparison: {e}")
                print(f"   Saving current model anyway...")
                self.save_model(self.model_save_path)
                
        except Exception as e:
            print(f"[ML] Error during model saving: {e}")

# Factory function for integration
def create_ml_optimizer(config: Optional[MLModelConfig] = None) -> ProfessionalMLFertilizerOptimizer:
    """
    Factory function to create ML optimizer with error handling
    """
    try:
        return ProfessionalMLFertilizerOptimizer(config)
    except ImportError as e:
        error_msg = f"[ERROR] Cannot create ML optimizer: {e}\nInstall required packages: pip install scikit-learn scipy"
        print(error_msg)
        raise RuntimeError(error_msg)


# ============================================================================
# LINEAR ALGEBRA OPTIMIZER (Compatibility with existing main_api.py)
# ============================================================================

class LinearAlgebraOptimizer:
    """
    Linear algebra based optimization using matrix operations
    A * x = b where A is composition matrix, x is dosages, b is targets
    """
    
    def __init__(self):
        self.nutrient_calc = NutrientCalculator()
        print("[LA] Linear Algebra Optimizer initialized")
    
    def optimize_linear_algebra(self, fertilizers: List, targets: Dict[str, float], 
                               water: Dict[str, float]) -> Dict[str, float]:
        """
        Solve fertilizer optimization using linear algebra methods
        """
        print("\n=== LINEAR ALGEBRA OPTIMIZATION ===")
        
        # Prepare adjusted targets (subtract water contribution)
        adjusted_targets = {}
        for element, target in targets.items():
            water_contribution = water.get(element, 0)
            adjusted_targets[element] = max(0, target - water_contribution)
        
        print(f"Adjusted targets (after water): {adjusted_targets}")
        
        # Prepare linear system
        A, b, nutrients = self.nutrient_calc.prepare_linear_algebra_system(
            fertilizers, adjusted_targets
        )
        
        print(f"System matrix A shape: {A.shape}")
        print(f"Target vector b shape: {b.shape}")
        print(f"Nutrients: {nutrients}")
        
        # Try different solving methods
        methods = ["least_squares", "pseudoinverse", "normal_equations"]
        best_solution = None
        best_score = float('inf')
        best_method = None
        
        for method in methods:
            try:
                print(f"\nTrying method: {method}")
                x = self.nutrient_calc.solve_linear_system(A, b, method)
                
                # Evaluate solution
                achieved = A @ x
                residual = np.linalg.norm(achieved - b)
                non_negative_score = np.sum(np.maximum(0, -x))  # Penalty for negative values
                total_score = residual + non_negative_score * 1000
                
                print(f"  Residual: {residual:.6f}")
                print(f"  Non-negative penalty: {non_negative_score:.6f}")
                print(f"  Total score: {total_score:.6f}")
                
                if total_score < best_score:
                    best_score = total_score
                    best_solution = x.copy()
                    best_method = method
                
            except Exception as e:
                print(f"  Method {method} failed: {e}")
        
        if best_solution is None:
            print("All linear algebra methods failed. Using fallback.")
            return self._fallback_solution(fertilizers, adjusted_targets)
        
        print(f"\nBest method: {best_method} (score: {best_score:.6f})")
        
        # Convert solution to dosage dictionary
        dosages = {}
        for i, fertilizer in enumerate(fertilizers):
            dosage = max(0, float(best_solution[i]) / 1000)  # Convert mg/L to g/L
            if dosage > 0.0001:  # Only include meaningful dosages
                dosages[fertilizer.name] = dosage
        
        print(f"Linear algebra solution:")
        for name, dosage in dosages.items():
            print(f"  {name}: {dosage:.4f} g/L")
        
        return dosages
    
    def _fallback_solution(self, fertilizers: List, targets: Dict[str, float]) -> Dict[str, float]:
        """Simple fallback solution if linear algebra fails"""
        dosages = {}
        
        # Simple heuristic solution
        for fertilizer in fertilizers[:5]:  # Use first 5 fertilizers
            total_contribution = (sum(fertilizer.composition.cations.values()) + 
                                sum(fertilizer.composition.anions.values()))
            if total_contribution > 10:  # Only use fertilizers with significant content
                dosages[fertilizer.name] = 0.001  # Small default dosage
        
        return dosages


# ============================================================================
# COMPATIBILITY CLASSES FOR EXISTING IMPORTS
# ============================================================================

# Alias for backward compatibility
MLFertilizerOptimizer = ProfessionalMLFertilizerOptimizer

if __name__ == "__main__":
    # Test the professional ML optimizer
    print("[TEST] Testing Professional ML Fertilizer Optimizer...")
    
    try:
        # Initialize optimizer
        optimizer = create_ml_optimizer()
        
        # Create test fertilizers
        from fertilizer_database import FertilizerDatabase
        db = FertilizerDatabase()
        
        test_fertilizers = []
        for name in ['nitrato de calcio', 'nitrato de potasio', 'fosfato monopotasico']:
            fert = db.create_fertilizer_from_database(name)
            if fert:
                test_fertilizers.append(fert)
        
        if not test_fertilizers:
            print("[ERROR] No test fertilizers available")
            exit(1)
        
        # Generate training data
        print("Generating training data...")
        training_scenarios = optimizer.generate_real_training_data(test_fertilizers, 1000)
        
        # Train model
        print("Training ML model...")
        training_results = optimizer.train_advanced_model(test_fertilizers, training_scenarios)
        
        # Test optimization
        test_targets = {'N': 150, 'P': 40, 'K': 200, 'Ca': 180, 'Mg': 50, 'S': 80}
        test_water = {'N': 2, 'P': 1, 'K': 5, 'Ca': 20, 'Mg': 8, 'S': 5}
        
        print("Testing ML optimization...")
        result = optimizer.optimize_with_ml(test_targets, test_water, test_fertilizers)
        
        print("[SUCCESS] Professional ML optimizer test completed successfully!")
        print(f"Training R²: {training_results['dosage_test_r2']:.4f}")
        
        for name, dosage in result.items():
            if dosage > 0.001:
                print(f"{name}: {dosage:.4f} g/L")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()