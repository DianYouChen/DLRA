import os
import glob

def listdir(path: str, rev: bool=True) -> str:
    """
    return the latest dir/file in one level

    """
    return sorted(os.listdir(path), reverse = rev)[0] # from large to small

def listAllFiles(path: str) -> list:
    # find all nc files
    return sorted(glob.glob(path + '/**/*.nc', recursive=True))

def listallfiles(path: str, rev: bool=False) -> str:
    """
    return the latest dir/file in one level

    """
    return sorted(os.listdir(path), reverse = rev)

def get_latest(dir:str) -> str:
    year = listdir(dir)
    yearMonth = listdir(os.path.join(dir, year))
    file = listdir(os.path.join(dir, year, yearMonth))
    return os.path.join(dir, year, yearMonth, file)

def get_latest_all(dir:str, rev: bool=False) -> str:
    year = listdir(dir)
    yearMonth = listdir(os.path.join(dir, year))
    files = listallfiles(os.path.join(dir, year, yearMonth))

    allpaths = []
    for n in range(len(files)):
        allpaths.append(os.path.join(inp_dir, year, yearMonth, files[n]))
    return allpaths

