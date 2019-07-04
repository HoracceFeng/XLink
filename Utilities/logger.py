import datetime
import logging

import sys

def get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S')  


def init_logger(log_file = None, log_path = None, log_level = logging.DEBUG, mode = 'w', stdout = True):
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_path is None:
        log_path = '~/temp/log/' 
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.log'

    log_file = os.path.join(log_path, log_file)
    print('log file path:' + log_file);
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logging.basicConfig(level = log_level,
                format= fmt,
                filename= os.path.abspath(log_file),
                filemode=mode)
    
    if stdout:
        console = logging.StreamHandler(stream = sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

#     console = logging.StreamHandler(stream = sys.stderr)
#     console.setLevel(log_level)
#     formatter = logging.Formatter(fmt)
#     console.setFormatter(formatter)
#     logging.getLogger('').addHandler(console)

