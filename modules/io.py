import os
import pandas as pd

def read_tsp_instance(filepath):
    """
    Reads a TSP instance from a CSV file.

    Parameters
    ----------
    filepath : str
        The path to the CSV file containing the TSP instance.

    Returns
    -------
    data : pandas.DataFrame
        A DataFrame containing the TSP instance data. If the file does not
        exist, returns `None`.

    Notes
    -----
    This function uses the `pandas` library to read the CSV file and the `os`
    library to check if the file exists. It drops the "X" and "Y" columns
    from the DataFrame, which are assumed to be coordinates and are not used
    in the TSP instance.
    """

    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        data.drop(columns=["X", "Y"], inplace=True)
        return data
    else:
        print(f"File {filepath} not found.")
        return None

def sort_filenames(filenames):
    to_sort = []

    for filename in filenames:
        size, b = filename.split('CTSP_')[1].split('C')
        clusters, c = b.split('_')
        instance = c.replace('.csv', '')
        
        to_sort.append((filename, int(size), int(clusters), int(instance)))
    sorted_tuples = sorted(to_sort, key=lambda x: (x[1], x[2], x[3]))

    return [(filename, clusters) for filename, _, clusters, _ in sorted_tuples]

def get_filenames(ROOT_PATH):
    """
    Returns a list of all filenames inside the given root path.

    Parameters
    ----------
    ROOT_PATH : str
        The path to the root directory.

    Returns
    -------
    filenames : list
        A list of full paths to all files inside the root directory.

    Notes
    -----
    This function uses the `os` module to list the contents of the root directory
    and filter out directories. It returns a list of full paths to all files.
    """

    filenames = []
    for f in os.listdir(ROOT_PATH):
        full_path = os.path.join(ROOT_PATH, f)
        if os.path.isfile(full_path):
            filenames.append(full_path)

    return sort_filenames(filenames)

