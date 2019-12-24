from pathlib import Path

def get_root_path() -> Path:
    ''' Returns project root folder. '''
    return Path.cwd().parent

def get_path(path_name: str) -> Path:
    ''' Returns data folder. '''
    return get_root_path() / path_name

if __name__ == '__main__':
    readme = get_root_path() / 'README.md'
    print(readme.is_file())
