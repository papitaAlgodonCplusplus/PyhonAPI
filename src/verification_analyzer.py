#!/usr/bin/env python3
"""
COMPLETE VERIFICATION ANALYZER MODULE
Solution verification and cost analysis with professional algorithms
"""

from typing import Dict, List, Any, Optional
import math

class SolutionVerifier:
    """Professional solution verification module"""

    def __init__(self):
        # Optimum nutrient ranges for hydroponic solutions (mg/L)
        self.nutrient_ranges = {
            # Macronutrients
            'N': {'min': 100, 'max': 200, 'optimal': 150, 'tolerance': 0.10},
            'P': {'min': 30, 'max': 60, 'optimal': 40, 'tolerance': 0.15},
            'K': {'min': 150, 'max': 350, 'optimal': 200, 'tolerance': 0.10},
            'Ca': {'min': 120, 'max': 220, 'optimal': 180, 'tolerance': 0.10},
            'Mg': {'min': 30, 'max': 80, 'optimal': 50, 'tolerance': 0.15},
            'S': {'min': 50, 'max': 120, 'optimal': 80, 'tolerance': 0.20},
            
            # Micronutrients
            'Fe': {'min': 1.0, 'max': 4.0, 'optimal': 2.0, 'tolerance': 0.25},
            'Mn': {'min': 0.3, 'max': 1.5, 'optimal': 0.5, 'tolerance': 0.30},
            'Zn': {'min': 0.1, 'max': 0.8, 'optimal': 0.3, 'tolerance': 0.30},
            'Cu': {'min': 0.05, 'max': 0.3, 'optimal': 0.1, 'tolerance': 0.40},
            'B': {'min': 0.2, 'max': 1.0, 'optimal': 0.5, 'tolerance': 0.30},
            'Mo': {'min': 0.01, 'max': 0.1, 'optimal': 0.05, 'tolerance': 0.50},
            
            # Other elements
            'Na': {'min': 0, 'max': 50, 'optimal': 0, 'tolerance': 1.0},
            'Cl': {'min': 0, 'max': 75, 'optimal': 0, 'tolerance': 1.0},
            'HCO3': {'min': 0, 'max': 100, 'optimal': 50, 'tolerance': 0.50}
        }
        
        # Critical ratios for ionic relationships
        self.ionic_ratios = {
            'K_Ca': {'min': 0.8, 'max': 1.5, 'optimal': 1.2},
            'Ca_Mg': {'min': 3.0, 'max': 8.0, 'optimal': 4.0},
            'K_Mg': {'min': 2.0, 'max': 6.0, 'optimal': 3.0},
            'N_K': {'min': 0.6, 'max': 1.2, 'optimal': 0.75}
        }

    def verify_concentrations(self, target_concentrations: Dict[str, float], 
                            final_concentrations: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Comprehensive verification of nutrient concentrations against targets
        """
        print(f"\nVERIFYING NUTRIENT CONCENTRATIONS")
        print(f"Target parameters: {len(target_concentrations)}")
        print(f"Final parameters: {len(final_concentrations)}")
        
        results = []

        for nutrient, target in target_concentrations.items():
            if nutrient in final_concentrations:
                final = final_concentrations[nutrient]
                deviation = final - target
                percentage_deviation = (abs(deviation) / target * 100) if target > 0 else 0

                # Get nutrient-specific ranges
                if nutrient in self.nutrient_ranges:
                    ranges = self.nutrient_ranges[nutrient]
                    tolerance = ranges['tolerance']
                    min_acceptable = target * (1 - tolerance)
                    max_acceptable = target * (1 + tolerance)
                    optimal = ranges['optimal']
                else:
                    # Default ranges for unknown nutrients
                    tolerance = 0.15
                    min_acceptable = target * (1 - tolerance)
                    max_acceptable = target * (1 + tolerance)
                    optimal = target

                # Determine status and color
                status, color, recommendation = self._evaluate_nutrient_status(
                    final, target, min_acceptable, max_acceptable, optimal, nutrient
                )

                result = {
                    'parameter': nutrient,
                    'target_value': round(target, 2),
                    'actual_value': round(final, 2),
                    'unit': 'mg/L',
                    'deviation': round(deviation, 2),
                    'percentage_deviation': round(percentage_deviation, 1),
                    'status': status,
                    'color': color,
                    'recommendation': recommendation,
                    'min_acceptable': round(min_acceptable, 2),
                    'max_acceptable': round(max_acceptable, 2),
                    'optimal_range': f"{ranges['min']}-{ranges['max']}" if nutrient in self.nutrient_ranges else f"{target*0.8:.0f}-{target*1.2:.0f}"
                }

                results.append(result)
                
                # Log significant deviations
                if percentage_deviation > 20:
                    print(f"  [WARNING]  {nutrient}: {percentage_deviation:.1f}% deviation ({status})")
                elif percentage_deviation > 10:
                    print(f"  [FORM] {nutrient}: {percentage_deviation:.1f}% deviation")

        print(f"[SUCCESS] Verification completed for {len(results)} nutrients")
        return results

    def _evaluate_nutrient_status(self, final: float, target: float, 
                                min_acceptable: float, max_acceptable: float, 
                                optimal: float, nutrient: str) -> tuple:
        """
        Evaluate nutrient status with professional criteria
        """
        # Excellent range (within 5% of optimal)
        if abs(final - optimal) / optimal <= 0.05:
            return "Excellent", "DarkGreen", f"{nutrient} concentration is excellent and within optimal range"
        
        # Good range (within acceptable limits)
        elif min_acceptable <= final <= max_acceptable:
            return "Good", "Green", f"{nutrient} concentration is within acceptable range"
        
        # Moderate deviation
        elif target * 0.7 <= final <= target * 1.3:
            if final > max_acceptable:
                return "High", "Orange", f"{nutrient} slightly elevated. Monitor for potential toxicity"
            else:
                return "Low", "Orange", f"{nutrient} slightly low. Consider increasing fertilizer"
        
        # Critical deviation
        else:
            if final > target * 1.3:
                severity = "Deviation High" if final > target * 1.5 else "High"
                color = "Red" if severity == "Deviation High" else "Orange"
                return severity, color, f"{nutrient} dangerously high. Reduce fertilizer immediately and dilute solution"
            else:
                severity = "Deviation Low" if final < target * 0.5 else "Low"
                color = "Red" if severity == "Deviation Low" else "Yellow"
                return severity, color, f"{nutrient} critically low. Increase fertilizer significantly"

    def verify_ionic_relationships(self, final_meq: Dict[str, float], 
                                 final_mmol: Dict[str, float], 
                                 final_mg: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Verify critical ionic relationships and ratios
        """
        print(f"\n[?][?]  VERIFYING IONIC RELATIONSHIPS")
        
        results = []

        # K:Ca ratio (meq/L basis)
        k_meq = final_meq.get('K', 0)
        ca_meq = final_meq.get('Ca', 0)
        
        if ca_meq > 0:
            k_ca_ratio = k_meq / ca_meq
            ratio_info = self.ionic_ratios['K_Ca']
            
            status, color, recommendation = self._evaluate_ratio_status(
                k_ca_ratio, ratio_info, "K:Ca", "meq/L ratio"
            )
            
            results.append({
                'relationship_name': 'K:Ca Ratio (meq/L)',
                'actual_ratio': round(k_ca_ratio, 2),
                'target_min': ratio_info['min'],
                'target_max': ratio_info['max'],
                'optimal': ratio_info['optimal'],
                'unit': 'meq/L ratio',
                'status': status,
                'color': color,
                'recommendation': recommendation
            })

        # Ca:Mg ratio (meq/L basis)
        mg_meq = final_meq.get('Mg', 0)
        
        if mg_meq > 0:
            ca_mg_ratio = ca_meq / mg_meq
            ratio_info = self.ionic_ratios['Ca_Mg']
            
            status, color, recommendation = self._evaluate_ratio_status(
                ca_mg_ratio, ratio_info, "Ca:Mg", "meq/L ratio"
            )
            
            results.append({
                'relationship_name': 'Ca:Mg Ratio (meq/L)',
                'actual_ratio': round(ca_mg_ratio, 2),
                'target_min': ratio_info['min'],
                'target_max': ratio_info['max'],
                'optimal': ratio_info['optimal'],
                'unit': 'meq/L ratio',
                'status': status,
                'color': color,
                'recommendation': recommendation
            })

        # K:Mg ratio (meq/L basis)
        if mg_meq > 0:
            k_mg_ratio = k_meq / mg_meq
            ratio_info = self.ionic_ratios['K_Mg']
            
            status, color, recommendation = self._evaluate_ratio_status(
                k_mg_ratio, ratio_info, "K:Mg", "meq/L ratio"
            )
            
            results.append({
                'relationship_name': 'K:Mg Ratio (meq/L)',
                'actual_ratio': round(k_mg_ratio, 2),
                'target_min': ratio_info['min'],
                'target_max': ratio_info['max'],
                'optimal': ratio_info['optimal'],
                'unit': 'meq/L ratio',
                'status': status,
                'color': color,
                'recommendation': recommendation
            })

        # N:K ratio (mg/L basis)
        n_mg = final_mg.get('N', 0)
        k_mg = final_mg.get('K', 0)
        
        if k_mg > 0:
            n_k_ratio = n_mg / k_mg
            ratio_info = self.ionic_ratios['N_K']
            
            status, color, recommendation = self._evaluate_ratio_status(
                n_k_ratio, ratio_info, "N:K", "mg/L ratio"
            )
            
            results.append({
                'relationship_name': 'N:K Ratio (mg/L)',
                'actual_ratio': round(n_k_ratio, 2),
                'target_min': ratio_info['min'],
                'target_max': ratio_info['max'],
                'optimal': ratio_info['optimal'],
                'unit': 'mg/L ratio',
                'status': status,
                'color': color,
                'recommendation': recommendation
            })

        print(f"[SUCCESS] Ionic relationship verification completed: {len(results)} ratios analyzed")
        return results

    def _evaluate_ratio_status(self, actual_ratio: float, ratio_info: Dict[str, float], 
                             ratio_name: str, unit: str) -> tuple:
        """
        Evaluate ionic ratio status
        """
        min_val = ratio_info['min']
        max_val = ratio_info['max']
        optimal = ratio_info['optimal']
        
        # Excellent (within 10% of optimal)
        if abs(actual_ratio - optimal) / optimal <= 0.10:
            return "Excellent", "DarkGreen", f"{ratio_name} ratio is optimal"
        
        # Good (within acceptable range)
        elif min_val <= actual_ratio <= max_val:
            return "Good", "Green", f"{ratio_name} ratio is within acceptable range"
        
        # Moderate imbalance
        elif min_val * 0.8 <= actual_ratio <= max_val * 1.2:
            return "Caution", "Orange", f"{ratio_name} ratio is outside optimal range but manageable"
        
        # Severe imbalance
        else:
            return "Imbalanced", "Red", f"{ratio_name} ratio is severely imbalanced and requires correction"

    def verify_ionic_balance(self, final_meq: Dict[str, float]) -> Dict[str, Any]:
        """
        Professional ionic balance verification with detailed analysis
        """
        print(f"\n[?][?]  VERIFYING IONIC BALANCE")
        
        # Define cations and anions
        cation_elements = ['Ca', 'K', 'Mg', 'Na', 'NH4', 'Fe', 'Mn', 'Zn', 'Cu']
        anion_elements = ['N', 'S', 'Cl', 'P', 'HCO3', 'B', 'Mo']
        
        # Calculate sums
        cation_sum = sum(final_meq.get(cation, 0) for cation in cation_elements)
        anion_sum = sum(final_meq.get(anion, 0) for anion in anion_elements)
        
        # Calculate balance metrics
        difference = abs(cation_sum - anion_sum)
        total_ions = cation_sum + anion_sum
        
        if total_ions > 0:
            difference_percentage = (difference / (total_ions / 2)) * 100
        else:
            difference_percentage = 0
        
        # Professional balance evaluation
        balance_status = self._evaluate_balance_status(difference_percentage)
        
        # Calculate acceptable tolerance
        tolerance = min(cation_sum, anion_sum) * 0.1  # 10% tolerance
        
        result = {
            'cation_sum': round(cation_sum, 3),
            'anion_sum': round(anion_sum, 3),
            'difference': round(difference, 3),
            'difference_percentage': round(difference_percentage, 2),
            'is_balanced': 1 if difference_percentage <= 10.0 else 0,
            'tolerance': round(tolerance, 3),
            'balance_status': balance_status['status'],
            'balance_color': balance_status['color'],
            'balance_recommendation': balance_status['recommendation'],
            'cation_distribution': self._calculate_ion_distribution(final_meq, cation_elements),
            'anion_distribution': self._calculate_ion_distribution(final_meq, anion_elements)
        }
        
        print(f"Cation sum: {cation_sum:.2f} meq/L")
        print(f"Anion sum: {anion_sum:.2f} meq/L")
        print(f"Balance error: {difference_percentage:.1f}% ({balance_status['status']})")
        
        return result

    def _evaluate_balance_status(self, difference_percentage: float) -> Dict[str, str]:
        """
        Evaluate ionic balance status with professional criteria
        """
        if difference_percentage <= 5.0:
            return {
                'status': 'Excellent',
                'color': 'DarkGreen',
                'recommendation': 'Ionic balance is excellent. No adjustment needed.'
            }
        elif difference_percentage <= 10.0:
            return {
                'status': 'Good',
                'color': 'Green',
                'recommendation': 'Ionic balance is acceptable. Minor adjustments may improve stability.'
            }
        elif difference_percentage <= 15.0:
            return {
                'status': 'Caution',
                'color': 'Orange',
                'recommendation': 'Ionic balance is outside optimal range. Review fertilizer ratios.'
            }
        elif difference_percentage <= 25.0:
            return {
                'status': 'Poor',
                'color': 'Red',
                'recommendation': 'Ionic balance is poor. Significant fertilizer adjustment required.'
            }
        else:
            return {
                'status': 'Critical',
                'color': 'DarkRed',
                'recommendation': 'Ionic balance is critically imbalanced. Complete formulation review needed.'
            }

    def _calculate_ion_distribution(self, final_meq: Dict[str, float], ion_list: List[str]) -> Dict[str, float]:
        """
        Calculate percentage distribution of ions
        """
        total = sum(final_meq.get(ion, 0) for ion in ion_list)
        
        if total > 0:
            return {ion: round((final_meq.get(ion, 0) / total) * 100, 1) for ion in ion_list}
        else:
            return {ion: 0.0 for ion in ion_list}

class CostAnalyzer:
    """Professional cost analysis module with market-based pricing"""

    def __init__(self):
        # Market-based fertilizer costs (CRC per kg)
        # Updated with realistic 2024 pricing
        self.fertilizer_costs = {
            # Acids - Precios basados en API de Costa Rica
            'Acido Nítrico': 10000.0,
            'Acido Nitrico': 10000.0,
            'Ácido Nítrico': 10000.0,
            'Acido Fosfórico': 10000.0,
            'Acido Fosforico': 10000.0,
            'Ácido Fosfórico': 10000.0,
            'Acido Sulfurico': 8500.0,  # Estimado basado en patrón de precios
            'Acido Sulfúrico': 8500.0,
            'Ácido Sulfúrico': 8500.0,
            
            # Nitrates - Precios basados en API + estimaciones realistas
            'Nitrato de calcio': 14875.0,  # Precio exacto de API
            'Nitrato de Calcio': 14875.0,
            'Calcium Nitrate': 14875.0,
            'Nitrato de potasio': 16000.0,  # Estimado (típicamente más caro que calcio)
            'Nitrato de Potasio': 16000.0,
            'Potassium Nitrate': 16000.0,
            'Nitrato de amonio': 9500.0,   # Estimado (más barato que calcio)
            'Nitrato de Amonio': 9500.0,
            'Ammonium Nitrate': 9500.0,
            'Nitrato de magnesio': 12500.0, # Estimado (intermedio)
            'Nitrato de Magnesio': 12500.0,
            'Magnesium Nitrate': 12500.0,
            
            # Sulfates - Basado en patrón de sulfato de amonio de API
            'Sulfato de amonio': 10520.25,  # Precio exacto de API
            'Sulfato de Amonio': 10520.25,
            'Ammonium Sulfate': 10520.25,
            'Sulfato de potasio': 18000.0,  # Estimado (típicamente caro)
            'Sulfato de Potasio': 18000.0,
            'Potassium Sulfate': 18000.0,
            'Sulfato de magnesio': 8500.0,  # Estimado (más barato)
            'Sulfato de Magnesio': 8500.0,
            'Magnesium Sulfate': 8500.0,
            'Sulfato de calcio': 7500.0,   # Estimado (barato)
            'Sulfato de Calcio': 7500.0,
            'Calcium Sulfate': 7500.0,
            
            # Phosphates - Típicamente caros en Costa Rica
            'Fosfato monopotasico': 25000.0,  # Estimado alto (premium)
            'Fosfato monopotásico': 25000.0,
            'Fosfato Monopotasico': 25000.0,
            'Fosfato Monopotásico': 25000.0,
            'Monopotassium Phosphate': 25000.0,
            'KH2PO4': 25000.0,
            'MKP': 25000.0,
            'Fosfato dipotasico': 23000.0,
            'Fosfato dipotásico': 23000.0,
            'Dipotassium Phosphate': 23000.0,
            'Fosfato monoamonico': 20000.0,  # MAP - común
            'Fosfato monoamónico': 20000.0,
            'Monoammonium Phosphate': 20000.0,
            'MAP': 20000.0,
            'Fosfato diamonico': 18000.0,    # DAP - más común
            'Fosfato diamónico': 18000.0,
            'Fosfato diamónico (DAP)': 18000.0,  # Variante con paréntesis
            'Diammonium Phosphate': 18000.0,
            'DAP': 18000.0,
            
            # Chlorides - Basado en precio de Cloruro de Potasio de API
            'Cloruro de calcio': 12000.0,   # Estimado (más barato que K)
            'Cloruro de Calcio': 12000.0,
            'Calcium Chloride': 12000.0,
            'Cloruro de potasio': 58125.0,  # Precio exacto de API
            'Cloruro de Potasio': 58125.0,  # Precio exacto de API
            'Potassium Chloride': 58125.0,
            'Cloruro de magnesio': 10000.0, # Estimado
            'Cloruro de Magnesio': 10000.0,
            'Magnesium Chloride': 10000.0,
            
            # Micronutrients - Típicamente muy caros en Costa Rica
            'Quelato de hierro': 75000.0,   # Premium quelatos
            'Quelato de Hierro': 75000.0,
            'Iron Chelate': 75000.0,
            'Fe-EDTA': 75000.0,
            'FeEDTA': 75000.0,
            'Sulfato de hierro': 15000.0,   # Más barato que quelatos
            'Sulfato de Hierro': 15000.0,
            'Iron Sulfate': 15000.0,
            'FeSO4': 15000.0,
            'Sulfato de manganeso': 18000.0,
            'Sulfato de Manganeso': 18000.0,
            'Manganese Sulfate': 18000.0,
            'MnSO4': 18000.0,
            'MnSO4.4H2O': 18000.0,
            'Sulfato de zinc': 22000.0,     # Más caro que manganeso
            'Sulfato de Zinc': 22000.0,
            'Zinc Sulfate': 22000.0,
            'ZnSO4': 22000.0,
            'ZnSO4.7H2O': 22000.0,
            'Sulfato de cobre': 28000.0,    # Más caro
            'Sulfato de Cobre': 28000.0,
            'Copper Sulfate': 28000.0,
            'CuSO4': 28000.0,
            'CuSO4.5H2O': 28000.0,
            'Sulfato de cobre (acidif)': 28000.0,  # Variante específica
            'Acido borico': 35000.0,        # Boro es caro
            'Ácido bórico': 35000.0,
            'Ácido Bórico': 35000.0,
            'Boric Acid': 35000.0,
            'H3BO3': 35000.0,
            'Molibdato de sodio': 95000.0,  # Molibdeno muy caro
            'Molibdato de Sodio': 95000.0,
            'Sodium Molybdate': 95000.0,
            'Na2MoO4': 95000.0,
            'Na2MoO4.2H2O': 95000.0
        }

        # ACTUALIZAR TAMBIÉN los regional_factors para Costa Rica:
        self.regional_factors = {
            'North America': 1.0,
            'Europe': 1.15,
            'Asia': 0.85,
            'Latin America': 1.0,      # Cambiar de 0.90 a 1.0 para Costa Rica
            'Costa Rica': 1.0,         # Agregar específico para Costa Rica
            'Default': 1.0
        }

    def calculate_solution_cost_with_api_data(self, fertilizer_amounts: Dict[str, float], 
                              concentrated_volume: float, 
                              diluted_volume: float,
                              region: str = 'Default') -> Dict[str, Any]:
        """
        Calculate comprehensive solution cost analysis
        """
        print(f"\n[MONEY] CALCULATING COST ANALYSIS")
        print(f"Fertilizer amounts: {len(fertilizer_amounts)} fertilizers")
        print(f"Concentrated volume: {concentrated_volume:.1f} L")
        print(f"Diluted volume: {diluted_volume:.1f} L")
        
        regional_factor = self.regional_factors.get(region, 1.0)
        
        cost_per_fertilizer = {}
        total_cost_concentrated = 0
        total_cost_diluted = 0
        
        # Calculate cost for each fertilizer
        for fertilizer, amount_kg in fertilizer_amounts.items():
            if amount_kg > 0:
                # Get cost per kg (try multiple name variations)
                cost_per_kg = self._get_fertilizer_cost(fertilizer) * regional_factor
                
                # Calculate costs
                cost_concentrated = amount_kg * cost_per_kg
                cost_diluted = cost_concentrated  # Same cost regardless of dilution
                
                cost_per_fertilizer[fertilizer] = {
                    'amount_kg': round(amount_kg, 4),
                    'cost_per_kg': round(cost_per_kg, 2),
                    'total_cost': round(cost_concentrated, 3),
                    'price_source': 'fallback'  # Default to fallback since we're using internal pricing
                }
                
                total_cost_concentrated += cost_concentrated
                total_cost_diluted += cost_diluted
                
                print(f"  {fertilizer}: {amount_kg:.3f} kg × ₡{cost_per_kg:.2f}/kg = ₡{cost_concentrated:.3f}")
        
        # Calculate percentage distribution
        percentage_per_fertilizer = {}
        if total_cost_concentrated > 0:
            for fertilizer, cost_info in cost_per_fertilizer.items():
                percentage = (cost_info['total_cost'] / total_cost_concentrated) * 100
                percentage_per_fertilizer[fertilizer] = round(percentage, 1)
        
        # Calculate per-unit costs
        cost_per_liter_concentrated = total_cost_concentrated / concentrated_volume if concentrated_volume > 0 else 0
        cost_per_liter_diluted = total_cost_diluted / diluted_volume if diluted_volume > 0 else 0
        cost_per_m3_diluted = cost_per_liter_diluted * 1000
        # Create simplified cost dictionary for backward compatibility
        simple_cost_per_fertilizer = {name: info['total_cost'] for name, info in cost_per_fertilizer.items()}
        
        # Calculate pricing summary
        total_fertilizers_used = len([f for f in cost_per_fertilizer.keys() if cost_per_fertilizer[f]['amount_kg'] > 0])
        api_prices_used = 0  # Default to 0 since we're using fallback prices
        fallback_prices_used = total_fertilizers_used
        api_price_coverage = 0.0  # Default to 0% since we're using fallback prices
        
        # Enhanced result structure to match expected format
        result = {
            'total_cost_crc': round(total_cost_concentrated, 3),
            'cost_per_liter_crc': round(cost_per_liter_diluted, 4),
            'cost_per_m3_crc': round(cost_per_m3_diluted, 2),
            'api_price_coverage_percent': api_price_coverage,
            'fertilizer_costs': simple_cost_per_fertilizer,
            'cost_percentages': percentage_per_fertilizer,
            'pricing_sources': {
                'api_prices_used': api_prices_used,
                'fallback_prices_used': fallback_prices_used
            },
            'cost_per_m3_diluted': round(cost_per_m3_diluted, 2),
            'cost_per_fertilizer': cost_per_fertilizer,
            'regional_factor': regional_factor,
            'region': region,
            
            # Legacy fields for backward compatibility
            'total_cost_concentrated': round(total_cost_concentrated, 3),
            'total_cost_diluted': round(total_cost_diluted, 3),
            'cost_per_liter_concentrated': round(cost_per_liter_concentrated, 4),
            'cost_per_liter_diluted': round(cost_per_liter_diluted, 4),
            'cost_per_fertilizer': simple_cost_per_fertilizer,
            'percentage_per_fertilizer': percentage_per_fertilizer,
            'detailed_costs': cost_per_fertilizer,
            
            # Pricing summary for detailed analysis
            'pricing_summary': {
                'api_price_coverage': api_price_coverage,
                'api_prices_used': api_prices_used,
                'fallback_prices_used': fallback_prices_used,
                'total_fertilizers_analyzed': total_fertilizers_used
            }
        }
        
        print(f"[MONEY] Total cost: ₡{total_cost_concentrated:.3f}")
        print(f"[WATER] Cost per liter: ₡{cost_per_liter_diluted:.4f}")
        
        return result

    def _get_fertilizer_cost(self, fertilizer_name: str) -> float:
        """
        Get fertilizer cost with intelligent name matching
        """
        # Try exact match first
        if fertilizer_name in self.fertilizer_costs:
            return self.fertilizer_costs[fertilizer_name]
        
        # Try case-insensitive match
        name_lower = fertilizer_name.lower()
        for cost_name, cost in self.fertilizer_costs.items():
            if cost_name.lower() == name_lower:
                return cost
        
        # Try partial matching
        for cost_name, cost in self.fertilizer_costs.items():
            if name_lower in cost_name.lower() or cost_name.lower() in name_lower:
                return cost
        
        # Try keyword matching for common fertilizers
        keyword_mapping = {
            'nitrato': 1.00,
            'sulfato': 0.80,
            'fosfato': 2.50,
            'cloruro': 1.20,
            'calcio': 0.80,
            'potasio': 1.40,
            'magnesio': 0.70,
            'hierro': 5.00,
            'zinc': 4.00,
            'cobre': 5.50,
            'manganeso': 3.50,
            'boro': 6.00,
            'molibdeno': 12.00,
            'acido': 1.50
        }
        
        for keyword, default_cost in keyword_mapping.items():
            if keyword in name_lower:
                print(f"    [MONEY] Using keyword-based cost for {fertilizer_name}: ₡{default_cost:.2f}/kg")
                return default_cost
        
        # Default cost for unknown fertilizers
        default_cost = 2.00
        print(f"    [WARNING]  Unknown fertilizer {fertilizer_name}, using default cost: ₡{default_cost:.2f}/kg")
        return default_cost