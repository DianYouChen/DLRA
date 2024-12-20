class Enum:
    @classmethod
    def name(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return key

    @classmethod
    def from_name(cls, enum_type_str):
        for key, value in cls.__dict__.items():
            if key == enum_type_str:
                return value
        assert f'{cls.__name__}:{enum_type_str} doesnot exist.'

class DataType:
    # NOTE: Every value has to be a power of 2.
    NoneAtAll = 0
    Rain = 1
    RainDiff = 2
    Hour = 4
    Month = 8
    Radar = 16
    Elevation = 32
    Slope = 64
    Aspect = 128

    @classmethod
    def all_data(cls):
        output = 0
        for key, value in cls.__dict__.items():
            if not isinstance(value, int):
                continue
            output += value
        return output

    @classmethod
    def count(cls, dtype):
        return cls.count2D(dtype) + cls.count1D(dtype)

    @classmethod
    def count1D(cls, dtype):
        dt = int((dtype & cls.Month) == cls.Month)
        return dt * 2 # (sin, cos)

    @classmethod
    def count2D(cls, dtype):
        rain = int(dtype & DataType.Rain == DataType.Rain)
        raindiff = int(dtype & DataType.RainDiff == DataType.RainDiff)
        radar = int(dtype & DataType.Radar == DataType.Radar)
        elevation = int(dtype & DataType.Elevation == DataType.Elevation)
        # slope = int(dtype & DataType.Slope == DataType.Slope)
        # aspect = int(dtype & DataType.Aspect == DataType.Aspect)
        return rain + raindiff + radar + elevation*3

    @classmethod
    def print(cls, dtype, prefix=''):
        for key, value in cls.__dict__.items():
            if value == cls.NoneAtAll or not isinstance(value, int):
                continue
            if (dtype & value) == value:
                print(f'[{prefix} Dtype]', key)
