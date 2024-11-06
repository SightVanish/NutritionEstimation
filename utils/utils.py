import os

def check_dirs(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def logtxt(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)
