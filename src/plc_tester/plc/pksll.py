from dataclasses import dataclass


@dataclass
class Result:
    alarms: int = 0


class PKSLL:

    alarms = 0

    def execute(self, signal):

        result = Result(alarms=0)

        return result
