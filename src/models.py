from pydantic import BaseModel
from typing import Dict, List, Optional, Any
# ==============================================================================
# LINEAR PROGRAMMING MODELS - ADD TO models.py
# ==============================================================================

from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np

class LinearProgrammingConfig(BaseModel):
    """Configuration for linear programming optimization"""
    solver_preference: str = "pulp"  # Options: "pulp", "scipy"
    max_individual_dosage: float = 5.0  # g/L
    max_total_dosage: float = 15.0  # g/L
    min_dosage_threshold: float = 0.001  # g/L
    deviation_weight: float = 1000.0  # Priority weight for minimizing deviations
    dosage_weight: float = 10.0  # Priority weight for minimizing dosages
    ionic_balance_weight: float = 500.0  # Priority weight for ionic balance
    max_solver_time: int = 60  # seconds
    tolerance: float = 1e-6
    apply_safety_caps: bool = True
    strict_caps: bool = True

class LinearProgrammingResult(BaseModel):
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
    

class NutrientDeviationAnalysis(BaseModel):
    """Detailed analysis of nutrient deviations"""
    nutrient: str
    target_mg_l: float
    achieved_mg_l: float
    deviation_percent: float
    status: str  # "Excellent", "Good", "Low", "High", "Deviation Low", "Deviation High"
    nutrient_type: str  # "Macro", "Micro"
    water_contribution: float
    fertilizer_contribution: float

class FertilizerUsageAnalysis(BaseModel):
    """Analysis of fertilizer usage efficiency"""
    fertilizer_name: str
    dosage_g_per_L: float
    dosage_ml_per_L: float
    utilization_percent: float
    primary_nutrients_supplied: List[str]
    cost_efficiency_score: Optional[float] = None

class LinearProgrammingReport(BaseModel):
    """Comprehensive linear programming optimization report"""
    optimization_summary: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    fertilizer_usage: Dict[str, Any]
    nutrient_analysis: List[NutrientDeviationAnalysis]
    fertilizer_analysis: List[FertilizerUsageAnalysis]
    recommendations: List[str]
    comparison_with_deterministic: Optional[Dict[str, Any]] = None

class OptimizationComparison(BaseModel):
    """Comparison between different optimization methods"""
    methods_compared: List[str]
    performance_metrics: Dict[str, Dict[str, float]]
    winner: str
    improvement_percent: float
    recommendation: str

class SafetyCapsResult(BaseModel):
    """Results from applying nutrient safety caps"""
    capped_concentrations: Dict[str, float]
    adjustments_made: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    total_adjustments: int
    strict_mode: bool
    safety_score: float
    summary: Dict[str, Any]

# ==============================================================================
# ENHANCED REQUEST MODEL FOR LINEAR PROGRAMMING
# ==============================================================================

class CalculationSettings(BaseModel):
    """Settings for calculation"""
    volume_liters: float = 1000.0  # Default volume in liters
    precision: int = 2  # Decimal precision for results
    units: str = "mg/L"  # Units for nutrient concentrations
    crop_phase: str = "General"  # Crop growth phase (e.g., "Vegetative", "Flowering")

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

class LinearProgrammingRequest(BaseModel):
    """Enhanced request model for linear programming optimization"""
    fertilizers: List[Fertilizer]
    target_concentrations: Dict[str, float]
    water_analysis: Dict[str, float]
    calculation_settings: CalculationSettings
    lp_config: LinearProgrammingConfig = LinearProgrammingConfig()
    compare_methods: bool = False  # Whether to compare with deterministic method
    generate_detailed_report: bool = True


class FertilizerDosage(BaseModel):
    dosage_ml_per_L: float
    dosage_g_per_L: float


class CalculationStatus(BaseModel):
    success: bool
    warnings: List[str]
    iterations: int
    convergence_error: float
        
# =============================================================================
# CORE FERTILIZER MODELS
# =============================================================================

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

# =============================================================================
# RESPONSE MODELS
# =============================================================================

class FertilizerDosage(BaseModel):
    dosage_ml_per_L: float
    dosage_g_per_L: float

class CalculationStatus(BaseModel):
    success: bool
    warnings: List[str]
    iterations: int
    convergence_error: float

# =============================================================================
# MACHINE LEARNING MODELS
# =============================================================================

class MLModelConfig(BaseModel):
    """Configuration for ML model training and optimization"""
    model_type: str = "RandomForest"  # Options: "RandomForest", "XGBoost"
    max_iterations: int = 100
    tolerance: float = 1e-6
    feature_scaling: bool = True
    n_estimators: int = 100
    max_depth: Optional[int] = None
    learning_rate: float = 0.1
    random_state: int = 42