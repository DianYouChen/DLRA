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
    wtDot = 2 # wind terrain dot product
    ERA5 = 4
    Month = 8
    Radar = 16
    Elevation = 32

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
        dt = int(dtype & cls.Month == cls.Month)
        return dt * 2 # (sin, cos)

    @classmethod
    def count2D(cls, dtype):
        rain = int(dtype & DataType.Rain == DataType.Rain)
        radar = int(dtype & DataType.Radar == DataType.Radar)
        elevation = int(dtype & DataType.Elevation == DataType.Elevation)
        wt = int(dtype & DataType.wtDot == DataType.wtDot)
        return rain + radar + elevation + wt
    
    @classmethod
    def countMinus(cls, dtype):
        # this function is for add_from_poni
        dt = int(dtype & cls.Month == cls.Month) * 2 #(sin, cos)
        wt = int(dtype & DataType.wtDot == DataType.wtDot)
        elevation = int(dtype & DataType.Elevation == DataType.Elevation)
        return dt + wt + elevation
    
    @classmethod
    def print(cls, dtype, prefix=''):
        for key, value in cls.__dict__.items():
            if value == cls.NoneAtAll or not isinstance(value, int):
                continue
            if (dtype & value) == value:
                print(f'[{prefix} Dtype]', key)