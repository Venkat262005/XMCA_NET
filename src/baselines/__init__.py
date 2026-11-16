"""Baseline modules for XMCA-Net comparative benchmarking."""

from src.baselines.common_schema import (
    BaselineDetectionOutput,
    BaselineLocalizationOutput,
    BaselineMitigationOutput,
    ContextSensitivityOutput,
)
from src.baselines.baselines import (
    SingleAxisDetector,
    SingleAxisNeuralClassifier,
    BaselineScore,
)
from src.baselines.halp_probe import HALPProbeDetector, HALPProbeMLP
from src.baselines.viser_detector import VisERDetector, VisERObjectScore
from src.baselines.halloc_detector import HalLocDetector, HalLocMultiHeadModule
from src.baselines.refind_adapter import REFINDDetector
from src.baselines.opera_generation_runner import (
    OPERAGenerationRunner,
    OperaOverTrustLogitsProcessor,
)
from src.baselines.realign_training import (
    ReAlignMitigationRunner,
    RDPOLoss,
)

__all__ = [
    "BaselineDetectionOutput",
    "BaselineLocalizationOutput",
    "BaselineMitigationOutput",
    "ContextSensitivityOutput",
    "SingleAxisDetector",
    "SingleAxisNeuralClassifier",
    "BaselineScore",
    "HALPProbeDetector",
    "HALPProbeMLP",
    "VisERDetector",
    "VisERObjectScore",
    "HalLocDetector",
    "HalLocMultiHeadModule",
    "REFINDDetector",
    "OPERAGenerationRunner",
    "OperaOverTrustLogitsProcessor",
    "ReAlignMitigationRunner",
    "RDPOLoss",
]
