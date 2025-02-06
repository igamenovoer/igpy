# logging helpers
import logging
from attrs import define, field

# use the logging templates as like this:
# logging.basicConfig(level=logging.INFO, format=LoggingTemplates.TimeNameLevel, datefmt=LoggingDateFormat.HourMinSec)
class LoggingTemplates:
    TimeNameLevel : str = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    TimeNameLevelMsec : str = "[%(asctime)s.%(msecs)d][%(name)s][%(levelname)s] %(message)s"
    
    TimeLevel : str = "[%(asctime)s][%(levelname)s] %(message)s"    
    TimeLevelMsec : str = "[%(asctime)s.%(msecs)d][%(levelname)s] %(message)s"
    
    NameLevel : str = "[%(name)s][%(levelname)s] %(message)s"
    
class LoggingDateFormat:
    HourMinSec : str = r"%H:%M:%S"
    YearMonthDayHourMinSec : str = r"%Y-%m-%d %H:%M:%S"
    YearMonthDay : str = r"%Y-%m-%d"
@define(kw_only=True)
class CustomLogger:
    m_logger_name : str = field(alias='logger_name')
    m_handler_name : str = field(alias='handler_name')
    m_formatter : logging.Formatter = field(default=None, alias='formatter')
    
    m_tag : str | None = field(alias='tag', default=None)
    m_handler : logging.Handler = field(init=False, default=None)
    m_logger : logging.Logger = field(init=False, default=None)
    
    class Levels:
        DEBUG = logging.DEBUG
        INFO = logging.INFO
        WARNING = logging.WARNING
        ERROR = logging.ERROR
        CRITICAL = logging.CRITICAL
    
    def __attrs_post_init__(self):
        self.m_logger = logging.getLogger(self.m_logger_name)
        if self.m_formatter is None:
            self.m_formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s][pid=%(process)d] %(message)s")
            
        self.m_handler = logging.StreamHandler()
        self.m_handler.name = self.m_handler_name
        self.m_handler.setFormatter(self.m_formatter)
        
        all_names = [x.name for x in self.m_logger.handlers]
        if self.m_handler.name not in all_names:
            self.m_logger.addHandler(self.m_handler)
    
    def set_logger_by_name(self, name : str):
        ''' set logger by name
        '''
        if self.m_logger is not None:
            self.m_logger.removeHandler(self.m_handler)
            
        self.m_logger = logging.getLogger(name)
        
        if self.m_handler not in self.m_logger.handlers:
            self.m_logger.addHandler(self.m_handler)
            
    def set_formatter(self, formatter : logging.Formatter):
        self.m_handler.setFormatter(formatter)
        
    def set_level(self, level : int):
        self.m_logger.setLevel(level)
        
    def set_tag(self, tag : str):
        self.m_tag = tag
        
    def make_message(self, msg) -> str:
        if self.m_tag is None or len(self.m_tag) == 0:
            out : str = f'{msg}'
        else:
            out : str = f'[tag={self.m_tag}] {msg}'
        return out
    
    def info(self, msg, *args, **kwargs):
        self.m_logger.info(self.make_message(msg), *args, **kwargs)
        
    def debug(self, msg, *args, **kwargs):
        self.m_logger.debug(self.make_message(msg), *args, **kwargs)
        
    def warning(self, msg, *args,  **kwargs):
        self.m_logger.warning(self.make_message(msg), *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        self.m_logger.error(self.make_message(msg), *args, **kwargs)

# copy from https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility
def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)
    
add_logging_level = addLoggingLevel