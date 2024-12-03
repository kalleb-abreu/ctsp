import os
import pandas as pd
from datetime import datetime


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
        df = pd.read_csv(filepath)
        data = df.drop(['X', 'Y'], axis=1)
        df_coords = df[['X', 'Y']]
        return data, df_coords
    else:
        print(f"File {filepath} not found.")
        return None


def sort_filenames(filenames):
    """
    Sort filenames based on problem size, number of clusters and instance number.

    Parameters
    ----------
    filenames : list
        List of filenames to sort.

    Returns
    -------
    list
        Sorted list of tuples containing (filename, number_of_clusters).
    """
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
    list
        A list of tuples containing (filepath, number_of_clusters) for all files
        inside the root directory, sorted by problem size and instance number.

    Notes
    -----
    This function uses the `os` module to list the contents of the root directory
    and filter out directories. The returned list is sorted using sort_filenames().
    """

    filenames = []
    for f in os.listdir(ROOT_PATH):
        full_path = os.path.join(ROOT_PATH, f)
        if os.path.isfile(full_path):
            filenames.append(full_path)

    return sort_filenames(filenames)


def get_results_file(results_dir, first_instance):
    """
    Get the path for saving results, either creating a new file or using existing.

    Parameters
    ----------
    results_dir : str
        Directory where results files are stored.
    first_instance : bool
        If True, creates a new results file. If False, uses the most recent one.

    Returns
    -------
    str
        Path to the results file.
    """
    if first_instance:
        start_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        return f"{results_dir}/ctsp_results_{start_date}.csv"
    else:
        files = [f for f in os.listdir(
            results_dir) if f.startswith('ctsp_results_')]
        return f"{results_dir}/{sorted(files)[-1]}"


def save_instance_results(results_file, instance_name, ctsp):
    """
    Save the results of a CTSP instance to a CSV file.

    Parameters
    ----------
    results_file : str
        Path to the CSV file where results should be saved.
    instance_name : str
        Name of the problem instance.
    ctsp : CTSP
        CTSP object containing the results to be saved.

    Notes
    -----
    If the results file already exists, appends the new results.
    If not, creates a new file with headers.
    """
    results_df = pd.DataFrame({
        "instance": [instance_name],
        "best_OF": [ctsp.best_cost],
        "best_SOL": [ctsp.best_solution],
        "build_time": [ctsp.build_time],
        "total_time": [ctsp.total_time],
    })

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, index=False)
