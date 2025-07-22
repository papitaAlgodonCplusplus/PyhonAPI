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