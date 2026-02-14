from typing import List
from enum import Enum

class QualityGrade(int, Enum):
    SAFE = 0
    OK = 1
    WARN = 2
    BAD = 3
    CRITICAL = 4

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


class Status(Enum):
    NORMAL = 0
    SORTING = 1
    SORTED = 2
    UPDATING = 3

class VerticesOrder(Enum):
    XYZ = 0
    XZY = 1
    YXZ = 2
    YZX = 3
    ZXY = 4
    ZYX = 5

class TrianglesOrder(Enum):
    P123 = 0
    P132 = 1
    P213 = 2
    P231 = 3
    P312 = 4
    P321 = 5

class IdOrder(Enum):
    ASC = 0
    DESC = 1