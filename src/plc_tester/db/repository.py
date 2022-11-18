from abc import ABC, abstractclassmethod
from datetime import date
from pandas import DataFrame


class Repository(ABC):
    @abstractclassmethod
    def getData(startTime: date, endTime: date) -> DataFrame:
        ...
