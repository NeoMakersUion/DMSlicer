# src/dmslicer/materials/__init__.py
from .enums import CompositionKind, BehaviorKind
from .interfaces import (
    AbstractMaterial,
    AbstractBehavior,
    AbstractDomainNeighborRule,
    AbstractMaterialPairRule,
    AbstractDomain,
)

from .materials import ConstantMaterial, GradientMaterial
from .behaviors import Behavior, IsolateBehavior, SourceBehavior, GradientBehavior
from .neighbors import DomainNeighborRule
from .material_pairs import MaterialPairRule
from .domains import Domain
from .rules import allow_interaction
from .reports import Issue, Severity
from .pipeline import safe_policy_run, PipelineResult

__all__ = [
    "CompositionKind", "BehaviorKind",
    "AbstractMaterial", "AbstractBehavior", "AbstractDomainNeighborRule", "AbstractMaterialPairRule", "AbstractDomain",
    "ConstantMaterial", "GradientMaterial",
    "Behavior", "IsolateBehavior", "SourceBehavior", "GradientBehavior",
    "DomainNeighborRule", "MaterialPairRule",
    "Domain",
    "allow_interaction",
    "Issue", "Severity",
    "safe_policy_run", "PipelineResult",
]
