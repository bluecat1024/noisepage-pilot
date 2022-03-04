"""
Contains logic of maintaining the separate file of knob sweeping.
First column is timestamp;
Following columns are explored knobs.
"""

import time
import sys

def init_knob_change_log(file_path, knob_sweep_space):
    """
    Given the knob exploration space, write the header line of knob log.

    Parameters:
    -----------
    file_path : str
        The file path of knob change log, within the result directory of experiment.
    knob_sweep_space : List[Tuple(List[str], List[Any])]
        The knob sweeping space defined in configuration.
    """
    with open(file_path, "w") as fw:
        headers = ["knob_apply_timestamp"]
        for knob_name, _ in knob_sweep_space:
            headers.append(knob_name[0])
        fw.write(','.join(headers) + '\n')

def insert_knob_change_log(file_path, dbms_config_path):
    """
    Retrieve the explored knob names and corresponding values
    from the DBMS configuration file, insert timestamp log. This routine is called
    immediately after Pilot applies a knob change to DBMS.

    Parameters:
    -----------
    file_path : str
        The file path of knob change log, within the result directory of experiment.
    dbms_config_path : str
        The path of DBMS configuration file generated by knob explore logic.
    """
    timestamp = time.time()
    # Parse configuration file to get all explored knob values.
    knob_value_map = dict()
    with open(dbms_config_path, 'r') as fr:
        for line in fr:
            tokens = line.strip().split()
            filtered_tokens = []
            for token in tokens:
                if token[0] == '#':
                    break
                filtered_tokens.append(token)

            if len(filtered_tokens) != 3 or filtered_tokens[1] != '=':
                continue

            knob_name, value = filtered_tokens[0], filtered_tokens[2]
            knob_value_map[knob_name] = value

    # Parse header of the log.
    with open(file_path, 'r') as fr:
        header = fr.readline().strip().split(',')
    # First column is timestamp.
    knob_explore_names = header[1:]
    for knob_name in knob_explore_names:
        if knob_name not in knob_value_map:
            raise ValueError(f"Knob {knob_name} not found in DBMS configuration.\nPlease check the knob exploration injection.\n")
    
    # Build the new line of the log.
    new_line_tokens = [str(timestamp)]
    for knob_name in knob_explore_names:
        new_line_tokens.append(str(knob_value_map[knob_name]))

    with open(file_path, 'a') as fa:
        fa.write(','.join(new_line_tokens) + '\n')

if __name__ == __main__:
    if len(sys.argv) != 3:
        raise ValueError(f"Wrong argument number: {len(sys.argv)} to 3.")
    file_path = sys.argv[1]
    dbms_config_path = sys.argv[2]
    insert_knob_change_log(file_path, dbms_config_path)
