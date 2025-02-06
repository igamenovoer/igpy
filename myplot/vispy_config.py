# configure vispy
# you must import this module and modify the content before importing vispy_plot

usable_backends = ['SDL2', 'PySide', 'PyQt5']
def set_candidate_backend(backends):
    ''' set a backend or a list of backends for vispy to use
    
    parameters
    -------------
    backends
        str or a list of str, set the candidate backends. 
        vispy will try the backend in the given order, and use the first one available.
        possible backends are 'pyqt5','pyqt4','sdl2','pyside'
        
    '''
    if type(backends) is str:
        backends = [backends]
        
    global usable_backends
    usable_backends = backends