from typing import List
from enum import Enum

class QualityGrade(str, Enum):
    SAFE = "SAFE"
    OK = "OK"
    WARN = "WARN"
    BAD = "BAD"
    CRITICAL = "CRITICAL"

    @staticmethod
    def classify(value: float, criteria: List[float], kind: str = "upper") -> "QualityGrade":
        if kind == "upper":
            if value <= criteria[0]:
                return QualityGrade.SAFE
            elif value <= criteria[1]:
                return QualityGrade.OK
            elif value <= criteria[2]:
                return QualityGrade.WARN
            elif value <= criteria[3]:
                return QualityGrade.BAD
            else:
                return QualityGrade.CRITICAL
        else:
            if value >= criteria[0]:
                return QualityGrade.SAFE
            elif value >= criteria[1]:
                return QualityGrade.OK
            elif value >= criteria[2]:
                return QualityGrade.WARN
            elif value >= criteria[3]:
                return QualityGrade.BAD
            else:
                return QualityGrade.CRITICAL
