from abc import ABC, abstractclassmethod
from datetime import date
from typing import List


class PLC(ABC):

    @abstractclassmethod
    def read(TagModel) -> TagModel:
        ...

    @abstractclassmethod
    def readAll(tags: List[TagModel]) -> List[TagModel]:
        ...

    @abstractclassmethod
    def write(TagModel) -> None:
        ...

    @abstractclassmethod
    def writeAll(tags: List[TagModel]) -> None:
        ...
