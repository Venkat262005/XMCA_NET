"""Configuration loader and schema validator for XMCA-Net.

Enforces strict execution settings and validates dataset/model specifications.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ExecutionConfig(BaseModel):
    strict_mode: bool = Field(default=True)
    allow_fallbacks: bool = Field(default=False)
    allow_synthetic_data: bool = Field(default=False)
    require_source_hashes: bool = Field(default=True)
    require_model_revision: bool = Field(default=True)
    kaggle_gpu_required: bool = Field(default=True)

    @field_validator("allow_fallbacks", "allow_synthetic_data")
    @classmethod
    def validate_no_fallbacks_or_synthetic(cls, v: bool, info) -> bool:
        if v:
            raise ValueError(f"Strict research mode prohibits '{info.field_name}=True'. Fallbacks and synthetic data are forbidden.")
        return v


class HardwareConfig(BaseModel):
    target_platform: str = "kaggle"
    expected_accelerator: str = "NVIDIA T4"
    expected_gpu_count: int = 2
    gpu_memory_gb_per_device: float = 15.0
    local_inference_permitted: bool = False


class EvaluationConfig(BaseModel):
    progressive_tiers: List[int] = Field(default=[100, 300, 600, 1000])
    matched_sample_ids_enforced: bool = True
    random_seed: int = 42

    @field_validator("progressive_tiers")
    @classmethod
    def validate_progressive_tiers(cls, tiers: List[int]) -> List[int]:
        expected = [100, 300, 600, 1000]
        if tiers != expected:
            raise ValueError(f"Evaluation tiers must be exactly {expected}, got {tiers}")
        return tiers


class PathsConfig(BaseModel):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    images_dir: str = "data/images"
    artifacts_dir: str = "artifacts"
    manifests_dir: str = "artifacts/manifests"
    corpora_dir: str = "artifacts/corpora"
    generations_dir: str = "artifacts/generations"
    features_dir: str = "artifacts/features"
    models_dir: str = "artifacts/models"
    reports_dir: str = "artifacts/reports"


class DatasetSourceConfig(BaseModel):
    repository: str
    data_file: str
    images_source: str
    expected_sha256: Optional[str] = None


class DatasetItemConfig(BaseModel):
    name: str
    full_name: str
    task_type: str
    splits: List[str]
    supported_tiers: List[int]
    has_attribution_ground_truth: bool
    attribution_focus: str
    source: DatasetSourceConfig
    required_fields: List[str]
    strategies: Optional[Dict[str, Any]] = None


class DatasetsConfig(BaseModel):
    datasets: Dict[str, DatasetItemConfig]

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, datasets: Dict[str, DatasetItemConfig]) -> Dict[str, DatasetItemConfig]:
        required = {"pope", "faithscore", "a_okvqa", "infoseek"}
        found = set(datasets.keys())
        if found != required:
            missing = required - found
            extra = found - required
            raise ValueError(
                f"Strict research requirement: exactly four datasets required: {required}. "
                f"Missing: {missing}, Extra: {extra}"
            )
        return datasets


class ModelItemConfig(BaseModel):
    family: str
    role: Optional[str] = "adapter_validation"
    model_id: str
    revision: Optional[str] = None
    resolved_full_sha: Optional[str] = None
    architecture: str
    dtype: str = "float16"
    device_map: str = "auto"
    max_memory_per_gpu: str = "14GiB"
    generation_params: Dict[str, Any] = Field(default_factory=dict)
    multimodal_processor: str
    notes: Optional[str] = None
    alternative_scale_test: Optional[Dict[str, Any]] = None


class ModelsConfig(BaseModel):
    models: Dict[str, ModelItemConfig]

    @field_validator("models")
    @classmethod
    def validate_models(cls, models: Dict[str, ModelItemConfig]) -> Dict[str, ModelItemConfig]:
        required = {"llava", "qwen2_vl", "internvl"}
        found = set(models.keys())
        if found != required:
            missing = required - found
            extra = found - required
            raise ValueError(
                f"Strict research requirement: exactly three VLM families required: {required}. "
                f"Missing: {missing}, Extra: {extra}"
            )
        return models

    def get_model(self, key: str) -> ModelItemConfig:
        if key not in self.models:
            raise KeyError(
                f"Explicit missing-model error: model '{key}' is not configured in models.yaml. "
                f"Available models are: {list(self.models.keys())}"
            )
        return self.models[key]


class ProjectConfig(BaseModel):
    name: str
    full_name: str
    version: str


class AppConfig(BaseModel):
    project: ProjectConfig
    execution: ExecutionConfig
    hardware: HardwareConfig
    evaluation: EvaluationConfig
    paths: PathsConfig


def load_yaml(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_app_config(config_path: Optional[Path] = None) -> AppConfig:
    if config_path is None:
        config_path = Path("configs/config.yaml")
    data = load_yaml(config_path)
    return AppConfig(**data)


def load_datasets_config(datasets_path: Optional[Path] = None) -> DatasetsConfig:
    if datasets_path is None:
        datasets_path = Path("configs/datasets.yaml")
    data = load_yaml(datasets_path)
    return DatasetsConfig(**data)


def load_models_config(models_path: Optional[Path] = None, require_revision: bool = True) -> ModelsConfig:
    if models_path is None:
        models_path = Path("configs/models.yaml")
    data = load_yaml(models_path)
    cfg = ModelsConfig(**data)
    
    if require_revision:
        for name, m in cfg.models.items():
            if not m.revision:
                raise ValueError(
                    f"Strict mode violation: model '{name}' does not have an explicit git commit revision pin."
                )
    return cfg

