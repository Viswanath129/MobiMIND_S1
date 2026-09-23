"""
MobiMIND Measurement Manifest & Provenance Schema
Defines rigorous scientific provenance categories for all reported metrics:
  - MEASURED: Directly observed from physical hardware / verified tensor execution.
  - DERIVED: Formally computed from verified MEASURED values (e.g., sample mean, standard error).
  - ESTIMATED: Model-based statistical approximation (must state model and bounds).
  - TARGET: Engineering goal or design specification.
  - UNKNOWN: Metric cannot be physically or defensibly verified on the current hardware/environment.
"""

from enum import Enum
from typing import Dict, Any, List

class ProvenanceType(str, Enum):
    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    ESTIMATED = "ESTIMATED"
    TARGET = "TARGET"
    UNKNOWN = "UNKNOWN"

def create_provenance_record(value: Any, prov: ProvenanceType, source: str, notes: str = "") -> Dict[str, Any]:
    return {
        "value": value,
        "provenance": prov.value,
        "source": source,
        "notes": notes
    }
