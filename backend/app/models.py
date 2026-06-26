from tortoise import fields
from tortoise.models import Model


class TrainingRun(Model):
    id = fields.UUIDField(pk=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    source_filename = fields.CharField(max_length=255)
    status = fields.CharField(max_length=32, default="running")
    error_message = fields.TextField(null=True)

    rows_before = fields.IntField(null=True)
    rows_after = fields.IntField(null=True)
    feature_count = fields.IntField(null=True)
    champion_model = fields.CharField(max_length=80, null=True)
    optimal_threshold = fields.FloatField(null=True)

    preprocessing_stats = fields.JSONField(default=dict)
    feature_engineering_stats = fields.JSONField(default=dict)
    model_selection_stats = fields.JSONField(default=dict)
    evaluation_stats = fields.JSONField(default=dict)

    model_artifact_path = fields.CharField(max_length=500, null=True)
    scaler_artifact_path = fields.CharField(max_length=500, null=True)
    feature_columns_artifact_path = fields.CharField(max_length=500, null=True)
    threshold_artifact_path = fields.CharField(max_length=500, null=True)

    class Meta:
        table = "training_runs"
        ordering = ["-created_at"]


class PredictionRecord(Model):
    id = fields.UUIDField(pk=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    training_run = fields.ForeignKeyField(
        "models.TrainingRun",
        related_name="predictions",
        on_delete=fields.CASCADE,
    )
    input_data = fields.JSONField(default=dict)
    probability = fields.FloatField()
    threshold = fields.FloatField()
    prediction = fields.IntField()
    label = fields.CharField(max_length=80)

    class Meta:
        table = "predictions"
        ordering = ["-created_at"]
