class CustomEnumBase:
    ''' base class of enum, with some utility functions
    '''
    @classmethod
    def get_enum_values(cls):
        ''' get all enum values
        '''
        out = []
        for x in dir(cls):
            if x.startswith('__'):
                continue
            if callable(getattr(cls, x)):
                continue
            out.append(getattr(cls,x))
        return out
    
    @classmethod
    def get_enum_dict(cls) -> dict[str, object]:
        ''' a dict mapping member name to value
        '''
        out = {}
        for x in dir(cls):
            if x.startswith('__'):
                continue
            if callable(getattr(cls, x)):
                continue
            out[x]=getattr(cls,x)
        return out