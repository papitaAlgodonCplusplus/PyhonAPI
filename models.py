from pydantic import BaseModel
from typing import Dict, List, Optional, Any

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

class SimpleResponse(BaseModel):
    fertilizer_dosages: Dict[str, FertilizerDosage]
    calculation_status: CalculationStatus
    pdf_report: Optional[Dict[str, str]] = None

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

class MLTrainingData(BaseModel):
    """Training data structure for ML model"""
    features: Dict[str, float]  # Input features (targets + water analysis + derived)
    targets: Dict[str, float]   # Target dosages for each fertilizer
    quality_metrics: Dict[str, float]  # Quality assessment of the solution

class MLPrediction(BaseModel):
    """ML model prediction results"""
    predicted_dosages: Dict[str, float]
    confidence_score: float
    ionic_balance_prediction: float
    estimated_deviation: float

class MLTrainingResults(BaseModel):
    """Results from ML model training"""
    model_type: str
    training_samples: int
    feature_count: int
    fertilizer_count: int
    train_mae_overall: float
    test_mae_overall: float
    train_scores: List[Dict[str, Any]]
    test_scores: List[Dict[str, Any]]
    feature_names: List[str]
    target_names: List[str]
    training_time_seconds: float
    model_ready: bool = True

# =============================================================================
# VERIFICATION AND ANALYSIS MODELS
# =============================================================================

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
    optimization_suggestions: List[str]

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

# =============================================================================
# ADVANCED RESPONSE MODELS
# =============================================================================

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
    pdf_report: Optional[Dict[str, Any]] = None

class OptimizationMethodComparison(BaseModel):
    """Comparison results between different optimization methods"""
    test_conditions: Dict[str, Any]
    results: Dict[str, Any]  # Method name -> results
    performance_comparison: Dict[str, Any]

class DatabaseInfo(BaseModel):
    """Fertilizer database information"""
    total_fertilizers: int
    fertilizers: List[Dict[str, Any]]
    fertilizers_by_type: Dict[str, int]
    validation_report: Dict[str, Any]
    database_status: str
    coverage: Dict[str, str]

# =============================================================================
# SWAGGER INTEGRATION MODELS
# =============================================================================

class SwaggerIntegrationMetadata(BaseModel):
    """Metadata for Swagger API integration"""
    data_source: str
    catalog_id: int
    phase_id: int
    water_id: int
    fertilizers_analyzed: int
    fertilizers_processed: int
    fertilizers_matched: int
    optimization_method: str
    calculation_timestamp: str

class SwaggerIntegrationResponse(BaseModel):
    """Complete response from Swagger integration"""
    integration_metadata: SwaggerIntegrationMetadata
    performance_metrics: Dict[str, Any]
    calculation_results: AdvancedFertilizerResponse
    data_sources: Dict[str, str]

# =============================================================================
# LINEAR ALGEBRA MODELS
# =============================================================================

class LinearAlgebraResult(BaseModel):
    """Results from linear algebra optimization"""
    method_used: str
    matrix_condition_number: float
    residual_norm: float
    solution_vector: List[float]
    convergence_info: Dict[str, Any]

class SystemMatrixInfo(BaseModel):
    """Information about the linear algebra system matrix"""
    matrix_shape: List[int]  # [rows, columns]
    condition_number: float
    rank: int
    elements_analyzed: List[str]
    fertilizers_analyzed: List[str]

# =============================================================================
# QUALITY ASSESSMENT MODELS
# =============================================================================

class SolutionQuality(BaseModel):
    """Overall solution quality assessment"""
    overall_score: int  # 0-100
    concentration_check: Dict[str, Any]
    ionic_balance_check: Dict[str, Any]
    ec_ph_check: Dict[str, Any]
    recommendations: List[str]

class NutrientEfficiency(BaseModel):
    """Nutrient use efficiency metrics"""
    nutrient_efficiency: Dict[str, float]  # Per element efficiency %
    overall_efficiency: float
    cost_effectiveness: float
    waste_analysis: Dict[str, Any]

# =============================================================================
# BATCH PROCESSING MODELS
# =============================================================================

class BatchCalculationRequest(BaseModel):
    """Request for batch calculations with multiple scenarios"""
    base_request: FertilizerRequest
    parameter_variations: List[Dict[str, Any]]
    comparison_metrics: List[str]

class BatchCalculationResult(BaseModel):
    """Results from batch calculations"""
    total_calculations: int
    successful_calculations: int
    results: List[AdvancedFertilizerResponse]
    comparison_summary: Dict[str, Any]
    best_solution: Optional[Dict[str, Any]] = None

# =============================================================================
# VALIDATION AND ERROR MODELS
# =============================================================================

class ValidationError(BaseModel):
    """Validation error details"""
    field: str
    message: str
    received_value: Any
    expected_type: str

class CalculationError(BaseModel):
    """Calculation error details"""
    error_type: str
    error_message: str
    error_context: Dict[str, Any]
    suggested_fix: Optional[str] = None

class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool
    data: Optional[Any] = None
    error: Optional[CalculationError] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str

# =============================================================================
# EXPORT/IMPORT MODELS
# =============================================================================

class ExportRequest(BaseModel):
    """Request for exporting calculation results"""
    format: str  # "pdf", "excel", "json", "csv"
    include_charts: bool = True
    include_recommendations: bool = True
    custom_title: Optional[str] = None

class ImportRequest(BaseModel):
    """Request for importing fertilizer data or calculations"""
    data_type: str  # "fertilizers", "water_analysis", "targets"
    format: str  # "json", "csv", "excel"
    data: Any
    validation_level: str = "strict"  # "strict", "loose", "none"

# =============================================================================
# REPORTING MODELS
# =============================================================================

class ReportConfiguration(BaseModel):
    """Configuration for report generation"""
    report_type: str  # "standard", "detailed", "summary", "comparison"
    include_sections: List[str]
    language: str = "en"
    units: str = "metric"
    precision: int = 3
    include_charts: bool = True
    include_recommendations: bool = True

class ReportMetadata(BaseModel):
    """Metadata for generated reports"""
    report_id: str
    generation_timestamp: str
    report_type: str
    file_path: str
    file_size_bytes: int
    page_count: Optional[int] = None
    sections_included: List[str]