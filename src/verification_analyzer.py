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
        print(f"\n🔍 VERIFYING NUTRIENT CONCENTRATIONS")
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
                    print(f"  ⚠️  {nutrient}: {percentage_deviation:.1f}% deviation ({status})")
                elif percentage_deviation > 10:
                    print(f"  📋 {nutrient}: {percentage_deviation:.1f}% deviation")

        print(f"✅ Verification completed for {len(results)} nutrients")
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
        print(f"\n⚖️  VERIFYING IONIC RELATIONSHIPS")
        
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

        print(f"✅ Ionic relationship verification completed: {len(results)} ratios analyzed")
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
        print(f"\n⚖️  VERIFYING IONIC BALANCE")
        
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

    def generate_verification_summary(self, verification_results: List[Dict], 
                                    ionic_relationships: List[Dict], 
                                    ionic_balance: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive verification summary
        """
        # Analyze verification results
        total_nutrients = len(verification_results)
        excellent_count = len([r for r in verification_results if r['status'] == 'Excellent'])
        good_count = len([r for r in verification_results if r['status'] == 'Good'])
        warning_count = len([r for r in verification_results if r['status'] in ['High', 'Low', 'Caution']])
        critical_count = len([r for r in verification_results if 'Critical' in r['status']])
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            excellent_count, good_count, warning_count, critical_count, total_nutrients
        )
        
        # Analyze ionic relationships
        balanced_ratios = len([r for r in ionic_relationships if r['status'] in ['Excellent', 'Good']])
        total_ratios = len(ionic_relationships)
        
        summary = {
            'overall_quality_score': quality_score,
            'nutrient_analysis': {
                'total_nutrients': total_nutrients,
                'excellent_nutrients': excellent_count,
                'good_nutrients': good_count,
                'warning_nutrients': warning_count,
                'critical_nutrients': critical_count,
                'success_rate': round((excellent_count + good_count) / total_nutrients * 100, 1) if total_nutrients > 0 else 0
            },
            'ionic_analysis': {
                'total_ratios': total_ratios,
                'balanced_ratios': balanced_ratios,
                'balance_percentage': round(balanced_ratios / total_ratios * 100, 1) if total_ratios > 0 else 0,
                'ionic_balance_status': ionic_balance['balance_status'],
                'ionic_balance_error': ionic_balance['difference_percentage']
            },
            'recommendations': self._generate_recommendations(
                verification_results, ionic_relationships, ionic_balance
            )
        }
        
        return summary

    def _calculate_quality_score(self, excellent: int, good: int, warning: int, 
                               critical: int, total: int) -> float:
        """
        Calculate overall solution quality score (0-100)
        """
        if total == 0:
            return 0
        
        score = (excellent * 100 + good * 85 + warning * 60 + critical * 20) / total
        return round(score, 1)

    def _generate_recommendations(self, verification_results: List[Dict], 
                                ionic_relationships: List[Dict], 
                                ionic_balance: Dict) -> List[str]:
        """
        Generate intelligent recommendations based on verification results
        """
        recommendations = []
        
        # Analyze critical nutrients
        critical_nutrients = [r for r in verification_results if 'Critical' in r['status']]
        high_nutrients = [r for r in verification_results if r['status'] == 'High']
        low_nutrients = [r for r in verification_results if r['status'] == 'Low']
        
        # Critical nutrient recommendations
        if critical_nutrients:
            for nutrient in critical_nutrients:
                if 'High' in nutrient['status']:
                    recommendations.append(f"URGENT: {nutrient['parameter']} is critically high ({nutrient['actual_value']} mg/L). Reduce corresponding fertilizer by 30-50% or dilute solution.")
                else:
                    recommendations.append(f"URGENT: {nutrient['parameter']} is critically low ({nutrient['actual_value']} mg/L). Increase corresponding fertilizer immediately.")
        
        # High nutrient recommendations
        if high_nutrients:
            high_elements = [n['parameter'] for n in high_nutrients]
            recommendations.append(f"Reduce fertilizers containing {', '.join(high_elements)} by 10-20% to prevent toxicity.")
        
        # Low nutrient recommendations
        if low_nutrients:
            low_elements = [n['parameter'] for n in low_nutrients]
            recommendations.append(f"Increase fertilizers containing {', '.join(low_elements)} by 15-25% to meet targets.")
        
        # Ionic balance recommendations
        balance_error = ionic_balance['difference_percentage']
        if balance_error > 15:
            if ionic_balance['cation_sum'] > ionic_balance['anion_sum']:
                recommendations.append("Ionic balance shows excess cations. Consider reducing Ca, K, or Mg sources and increasing anion sources.")
            else:
                recommendations.append("Ionic balance shows excess anions. Consider reducing N, P, or S sources and increasing cation sources.")
        
        # Ionic ratio recommendations
        imbalanced_ratios = [r for r in ionic_relationships if r['status'] in ['Caution', 'Imbalanced']]
        for ratio in imbalanced_ratios:
            if 'K:Ca' in ratio['relationship_name']:
                if ratio['actual_ratio'] > ratio['target_max']:
                    recommendations.append("K:Ca ratio is too high. Reduce potassium fertilizers or increase calcium fertilizers.")
                else:
                    recommendations.append("K:Ca ratio is too low. Increase potassium fertilizers or reduce calcium fertilizers.")
        
        # General optimization recommendations
        if balance_error <= 10 and not critical_nutrients:
            recommendations.append("Solution composition is well-balanced. Consider fine-tuning for specific crop requirements.")
        
        # EC and pH recommendations
        if ionic_balance['cation_sum'] + ionic_balance['anion_sum'] > 25:
            recommendations.append("Total ion concentration is high (EC >2.5). Consider dilution for sensitive crops.")
        elif ionic_balance['cation_sum'] + ionic_balance['anion_sum'] < 10:
            recommendations.append("Total ion concentration is low (EC <1.0). Increase overall fertilizer concentration.")
        
        return recommendations[:10]  # Limit to top 10 recommendations


class CostAnalyzer:
    """Professional cost analysis module with market-based pricing"""

    def __init__(self):
        # Market-based fertilizer costs (USD per kg)
        # Updated with realistic 2024 pricing
        self.fertilizer_costs = {
            # Acids
            'Acido Nítrico': 1.20,
            'Acido Nitrico': 1.20,
            'Ácido Nítrico': 1.20,
            'Acido Fosfórico': 1.80,
            'Acido Fosforico': 1.80,
            'Ácido Fosfórico': 1.80,
            'Acido Sulfurico': 0.90,
            'Acido Sulfúrico': 0.90,
            'Ácido Sulfúrico': 0.90,
            
            # Nitrates
            'Nitrato de calcio': 0.85,
            'Nitrato de Calcio': 0.85,
            'Calcium Nitrate': 0.85,
            'Nitrato de potasio': 1.30,
            'Nitrato de Potasio': 1.30,
            'Potassium Nitrate': 1.30,
            'Nitrato de amonio': 0.55,
            'Nitrato de Amonio': 0.55,
            'Ammonium Nitrate': 0.55,
            'Nitrato de magnesio': 1.10,
            'Nitrato de Magnesio': 1.10,
            'Magnesium Nitrate': 1.10,
            
            # Sulfates
            'Sulfato de amonio': 0.45,
            'Sulfato de Amonio': 0.45,
            'Ammonium Sulfate': 0.45,
            'Sulfato de potasio': 1.60,
            'Sulfato de Potasio': 1.60,
            'Potassium Sulfate': 1.60,
            'Sulfato de magnesio': 0.65,
            'Sulfato de Magnesio': 0.65,
            'Magnesium Sulfate': 0.65,
            'Sulfato de calcio': 0.40,
            'Sulfato de Calcio': 0.40,
            'Calcium Sulfate': 0.40,
            
            # Phosphates
            'Fosfato monopotasico': 2.80,
            'Fosfato monopotásico': 2.80,
            'Fosfato Monopotasico': 2.80,
            'Fosfato Monopotásico': 2.80,
            'Monopotassium Phosphate': 2.80,
            'KH2PO4': 2.80,
            'MKP': 2.80,
            'Fosfato dipotasico': 2.60,
            'Fosfato dipotásico': 2.60,
            'Dipotassium Phosphate': 2.60,
            'Fosfato monoamonico': 1.80,
            'Fosfato monoamónico': 1.80,
            'Monoammonium Phosphate': 1.80,
            'MAP': 1.80,
            'Fosfato diamonico': 1.70,
            'Fosfato diamónico': 1.70,
            'Diammonium Phosphate': 1.70,
            'DAP': 1.70,
            
            # Chlorides
            'Cloruro de calcio': 0.75,
            'Cloruro de Calcio': 0.75,
            'Calcium Chloride': 0.75,
            'Cloruro de potasio': 1.40,
            'Cloruro de Potasio': 1.40,
            'Potassium Chloride': 1.40,
            'Cloruro de magnesio': 0.80,
            'Cloruro de Magnesio': 0.80,
            'Magnesium Chloride': 0.80,
            
            # Micronutrients (typically more expensive)
            'Quelato de hierro': 8.50,
            'Quelato de Hierro': 8.50,
            'Iron Chelate': 8.50,
            'Fe-EDTA': 8.50,
            'FeEDTA': 8.50,
            'Sulfato de hierro': 2.20,
            'Sulfato de Hierro': 2.20,
            'Iron Sulfate': 2.20,
            'FeSO4': 2.20,
            'Sulfato de manganeso': 3.80,
            'Sulfato de Manganeso': 3.80,
            'Manganese Sulfate': 3.80,
            'MnSO4': 3.80,
            'MnSO4.4H2O': 3.80,
            'Sulfato de zinc': 4.20,
            'Sulfato de Zinc': 4.20,
            'Zinc Sulfate': 4.20,
            'ZnSO4': 4.20,
            'ZnSO4.7H2O': 4.20,
            'Sulfato de cobre': 5.60,
            'Sulfato de Cobre': 5.60,
            'Copper Sulfate': 5.60,
            'CuSO4': 5.60,
            'CuSO4.5H2O': 5.60,
            'Acido borico': 6.40,
            'Ácido bórico': 6.40,
            'Ácido Bórico': 6.40,
            'Boric Acid': 6.40,
            'H3BO3': 6.40,
            'Molibdato de sodio': 12.50,
            'Molibdato de Sodio': 12.50,
            'Sodium Molybdate': 12.50,
            'Na2MoO4': 12.50,
            'Na2MoO4.2H2O': 12.50
        }
        
        # Regional price adjustment factors
        self.regional_factors = {
            'North America': 1.0,
            'Europe': 1.15,
            'Asia': 0.85,
            'Latin America': 0.90,
            'Default': 1.0
        }

    def calculate_solution_cost(self, fertilizer_amounts: Dict[str, float], 
                              concentrated_volume: float, 
                              diluted_volume: float,
                              region: str = 'Default') -> Dict[str, Any]:
        """
        Calculate comprehensive solution cost analysis
        """
        print(f"\n💰 CALCULATING COST ANALYSIS")
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
                    'total_cost': round(cost_concentrated, 3)
                }
                
                total_cost_concentrated += cost_concentrated
                total_cost_diluted += cost_diluted
                
                print(f"  {fertilizer}: {amount_kg:.3f} kg × ${cost_per_kg:.2f}/kg = ${cost_concentrated:.3f}")
        
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
        
        result = {
            'total_cost_concentrated': round(total_cost_concentrated, 3),
            'total_cost_diluted': round(total_cost_diluted, 3),
            'cost_per_liter_concentrated': round(cost_per_liter_concentrated, 4),
            'cost_per_liter_diluted': round(cost_per_liter_diluted, 4),
            'cost_per_m3_diluted': round(cost_per_m3_diluted, 2),
            'cost_per_fertilizer': simple_cost_per_fertilizer,  # Simplified for compatibility
            'percentage_per_fertilizer': percentage_per_fertilizer,
            'detailed_costs': cost_per_fertilizer,  # Detailed cost breakdown
            'regional_factor': regional_factor,
            'region': region
        }
        
        print(f"💰 Total cost: ${total_cost_concentrated:.3f}")
        print(f"💧 Cost per liter: ${cost_per_liter_diluted:.4f}")
        
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
                print(f"    💰 Using keyword-based cost for {fertilizer_name}: ${default_cost:.2f}/kg")
                return default_cost
        
        # Default cost for unknown fertilizers
        default_cost = 2.00
        print(f"    ⚠️  Unknown fertilizer {fertilizer_name}, using default cost: ${default_cost:.2f}/kg")
        return default_cost