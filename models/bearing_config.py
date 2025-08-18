from dataclasses import dataclass

@dataclass
class BearingConfig:
    test_1_bearings: int = 4
    test_1_sensors_per_bearing: int = 2
    test_2_bearings: int = 4
    test_2_sensors_per_bearing: int = 1
    test_3_bearings: int = 4
    test_3_sensors_per_bearing: int = 1
    sample_rate: int = 20000 # Sample frequency in Hz
    sample_duration: int = 1 # No. seconds per file
    test_1_early_interval: int = 5 # First 44 samples of test one occured every 10 minutes
    test_1_late_interval: int = 10 # all following tests occured every 10 minutes
    test_1_early_file_cutoff: int = 44
    standard_interval: int = 10