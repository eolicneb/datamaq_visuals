import pytest

from src.core.meassurements.average import *


def test_averaged_measurement():
    measurement = AveragedMeasurement[int]("test", allowed_deviation=2,
                                           buffer_size=100, cast=int)
    for i in range(30):
        measurement.update(i**(1.2 + .1*(i>20)))
    print(measurement.value())
    assert measurement.value()
