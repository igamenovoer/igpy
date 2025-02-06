# bentoML utilities
import bentoml as bml
import inspect

class RunnerUtils:
    @classmethod
    def get_method_signature(cls, runner : bml.Runner, method_name : str) -> inspect.Signature:
        ''' get the method signature of the given method name of the given runner.
        
        parameters
        ------------
        runner : bml.Runner
            the runner to inspect
        method_name : str
            the method name to inspect
        
        return
        ---------
        output : str
            the method signature
        '''
        
        if method_name == 'run':
            target_method_name = '__call__'
        else:
            target_method_name = method_name
            
        method = getattr(runner._runner_handle._runnable.model, target_method_name)
        sig = inspect.signature(method)
        return sig
    
    @classmethod
    def get_method_signatures(cls, runner : bml.Runner) -> dict[str, inspect.Signature]:
        ''' get the method signatures of the given runner.
        '''
        out = {}
        for m in runner.runner_methods:
            out[m.name] = cls.get_method_signature(runner, m.name)
        return out
    
    @classmethod
    def get_underlying_model(cls, runner : bml.Runner):
        ''' get the underlying model of the given runner.
        
        parameters
        ------------
        runner : bml.Runner
            the runner to inspect
        
        return
        ---------
        output : object
            the underlying model
        '''
        return runner._runner_handle._runnable.model