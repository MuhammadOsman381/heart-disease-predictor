from typing import Any

from pydantic import BaseModel, Field


class PreprocessingStats(BaseModel):
    rows_before: int | None = None
    rows_after: int | None = None
    duplicates_removed: int | None = None
    missing_values_before_drop: dict[str, int] = Field(default_factory=dict)
    target_distribution_before: dict[str, int] = Field(default_factory=dict)
    target_distribution_after: dict[str, int] = Field(default_factory=dict)
    continuous_columns_scaled: list[str] = Field(default_factory=list)
    categorical_columns_encoded: list[str] = Field(default_factory=list)


class FeatureEngineeringStats(BaseModel):
    engineered_features: list[str] = Field(default_factory=list)
    feature_count: int | None = None
    train_shape: list[int] = Field(default_factory=list)
    test_shape: list[int] = Field(default_factory=list)
    class_balance_train: dict[str, float] = Field(default_factory=dict)
    class_balance_test: dict[str, float] = Field(default_factory=dict)


class ModelMetricStats(BaseModel):
    mean_recall: float | None = None
    std_recall: float | None = None
    mean_accuracy: float | None = None
    mean_roc_auc: float | None = None
    weighted_score: float | None = None


class ModelSelectionStats(BaseModel):
    baseline_results: dict[str, ModelMetricStats] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    champion_model: str | None = None
    final_model_params: dict[str, Any] = Field(default_factory=dict)


class ConfusionMatrixStats(BaseModel):
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0


class ThresholdStats(BaseModel):
    threshold: float | None = None
    recall: float | None = None
    precision: float | None = None
    f1: float | None = None


class EvaluationStats(BaseModel):
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    specificity: float | None = None
    confusion_matrix: ConfusionMatrixStats = Field(default_factory=ConfusionMatrixStats)
    classification_report: dict[str, Any] = Field(default_factory=dict)
    optimal_threshold_result: ThresholdStats = Field(default_factory=ThresholdStats)


class TrainingRunResponse(BaseModel):
    id: str
    created_at: str | None = None
    updated_at: str | None = None
    source_filename: str
    status: str
    error_message: str | None = None

    rows_before: int | None = None
    rows_after: int | None = None
    feature_count: int | None = None
    champion_model: str | None = None
    optimal_threshold: float | None = None

    preprocessing_stats: PreprocessingStats = Field(default_factory=PreprocessingStats)
    feature_engineering_stats: FeatureEngineeringStats = Field(default_factory=FeatureEngineeringStats)
    model_selection_stats: ModelSelectionStats = Field(default_factory=ModelSelectionStats)
    evaluation_stats: EvaluationStats = Field(default_factory=EvaluationStats)

    model_artifact_path: str | None = None
    scaler_artifact_path: str | None = None
    feature_columns_artifact_path: str | None = None
    threshold_artifact_path: str | None = None


class PipelineStatsResponse(BaseModel):
    run_id: str
    preprocessing: PreprocessingStats
    feature_engineering: FeatureEngineeringStats
    model_selection: ModelSelectionStats
    evaluation: EvaluationStats


class PatientInput(BaseModel):
    run_id: str | None = None
    age: float
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: float
    chol: float
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


class PredictionResponse(BaseModel):
    id: str | None = None
    run_id: str
    created_at: str | None = None
    probability: float
    threshold: float
    prediction: int
    label: str


class PredictionRecordResponse(PredictionResponse):
    input_data: dict[str, Any] = Field(default_factory=dict)


class PredictionAnalysisResponse(BaseModel):
    prediction_id: str
    system_prompt: str
    summary: str
    possible_reasons: list[str] = Field(default_factory=list)
    suggested_steps: list[str] = Field(default_factory=list)
    safety_note: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    prediction_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    prediction_id: str
    system_prompt: str
    reply: str
