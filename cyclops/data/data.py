import importlib.resources
import sys

def get_data_file_path(filename: str) -> str:
    """
    Get the full, absolute path to a data file inside the package.

    Parameters
    ----------
    filename : str
        The name of the file within the 'cyclops/data' directory.

    Returns
    -------
    str
        The absolute path to the data file.
    """
    
    # Python 3.9+ introduced the .files() API, which is cleaner.
    # __name__ here resolves to "cyclops.data"
    if sys.version_info >= (3, 9):
        try:
            with importlib.resources.files(__package__) as data_dir:
                file_path = data_dir / filename
                if not file_path.is_file():
                    raise FileNotFoundError(
                        f"Data file '{filename}' not found in {data_dir}"
                    )
                return str(file_path)
        except Exception as e:
            # Fallback in case of unexpected issues
            print(f"Warning: Using fallback for data file path. Error: {e}")

    # Fallback for Python 3.8 or if the .files() API fails
    with importlib.resources.path(__package__, filename) as path:
        return str(path)