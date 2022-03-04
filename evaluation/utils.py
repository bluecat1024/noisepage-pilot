import numpy as np
import xml.etree.ElementTree as ET
from psycopg import cursor
import yaml

def param_sweep_space(ps_dist):
    '''Construct parameter sweep space from configuration.
    
    Parameters:
    ------------
    ps_dist : Dict[str, List[Any]]
        Contains parameter name to candidate value lists.

    Returns:
    ---------
    ps_space : List[Tuple(List[str], List[Any])]
        Return parameter sweeping space as a list of tuples of pairs of list.
        Parse name as name level, handle linear range scales.
    '''
    ps_space = []
    assert ps_dist is not None and len(ps_dist) > 0, 'Parameter space should not be empty.\nCheck the configuration file.'
    for name in ps_dist:
        val_list = ps_dist[name]
        assert val_list is not None and len(val_list) > 0, 'Parameter space should not be empty.\nCheck the configuration file.'
        
        xml_level = list(name.strip().split('.'))

        # Linear scale range
        if (val_list[0] == '.range'):
            def convert_range(range_list):
                '''Convert a list starting with .range to linear range.

                Parameters:
                ------------
                range_list : List[Any]
                The range statement. First should be .range; should be 3 or 4 in length.
                
                Returns:
                -----------
                values : List[Num]
                The list of numeric values generated.
                '''
                LENGTH_NO_STEP = 3
                LENGTH_WITH_STEP = 4
                assert len(range_list) in (LENGTH_NO_STEP, LENGTH_WITH_STEP), 'Incorrect argument number for linear scale.\nCheck the configuration file.'
                if (len(range_list) == LENGTH_NO_STEP):
                    start, end, step = range_list[1], range_list[2], 1
                else:
                    start, end, step = range_list[1:]

                values = np.arange(start, end, step).tolist()
                return values

            val_list = convert_range(val_list)

        ps_space.append((xml_level, val_list))

    return ps_space

def inject_knob_exploration(file_path, knobs):
    """
    Inject and re-write Postgres file configurations given knobs.

    Parameters:
    -----------
    file_path : str
        Configuration file path to inject.
    knobs : List[Tuple(List[str], Any)]
        The list of parameter names and values to inject.
    """

    knob_value_map = dict()
    for name, value in knobs:
        knob_value_map[name[0]] = value

    new_line_list = []
    with open(file_path, 'r') as fr:
        line_list = list(fr.readlines())
        # Process the DBMS configuration file.
        for line in line_list:
            tokens = line.strip().split()
            filtered_tokens = []
            for token in tokens:
                if token[0] == '#':
                    break
                filtered_tokens.append(token)

            if len(filtered_tokens) != 3 or filtered_tokens[1] != '=':
                new_line_list.append(line)
            else:
                knob_name = filtered_tokens[0]
                # Replace value if already in configuration.
                if knob_name in knob_value_map:
                    new_line = f"{knob_name} = {knob_value_map[knob_name]}\n"
                    del knob_value_map[knob_name]
                else:
                    new_line = line

                new_line_list.append(new_line)
    
    # Append the knob settings to tail of the new file.
    for knob_name, value in knob_value_map.items():
        new_line = f"{knob_name} = {value}\n"
        new_line_list.append(new_line)

    # Flush to the new file.
    with open(file_path, 'w') as fw:
        for line in new_line_list:
            fw.write(line)
    

def inject_param_xml(file_path, parameters):
    '''Inject and re-write XML file with given parameters.

    Parameters:
    -----------
    file_path : str
        XML file path to inject.
    parameters : List[Tuple(List[str], Any)]
        The list of parameter names and values to inject.
    '''
    conf_etree = ET.parse(file_path)
    root = conf_etree.getroot()
    for name_level, val in parameters:
        cursor = root
        # Traverse XML name levels.
        for key in name_level:
            cursor = cursor.find(key)
            if cursor is None:
                break

        assert cursor is not None, 'Fail to inject parameter in conf file,' + str(name_level) + '\nCheck the format of target XML file.'
        cursor.text = str(val)

    conf_etree.write(file_path)

def parameter_sweep(ps_space, f, closure=None):
    '''Recursive calling routine on parameter space.

    Parameters:
    ------------
    ps_space : List[Tuple(List[str], List[Any])]
        Parameter space to sweep. each element is parameter name level + value list.
    f : (List[Tuple(List[str], Any)], Dict[str, Any])->Any
        Callback function to be executed in the sweep, takes parameter combination
        and closure dict.
    closure : Dict[str, Any]
        Closure environment passed from caller.

    Returns:
    ------------
    return_values : List[Any]
        A list containing the return value of all invocations of callback.
    '''
    assert(len(ps_space) > 0), 'Parameter space should not be empty.\nCheck the configuration file.'

    if closure is None:
        closure = {}

    # The return value list.
    return_values = []
    # Maintain the traverse states. Initial to the first value on level 0.
    cursor_stack = [-1]
    parameters = []
    while len(cursor_stack) > 0:
        depth = len(cursor_stack) - 1
        cursor_stack[depth] += 1
        name_level, val_list = ps_space[depth]
        if (cursor_stack[depth] >= len(val_list)):
            # Backtrace.
            del cursor_stack[-1]
            if len(parameters) > 0:
                del parameters[-1]
            continue

        name_level = ps_space[depth][0]
        value = ps_space[depth][1][cursor_stack[depth]]
        parameters.append((name_level, value))

        if len(cursor_stack) >= len(ps_space):
            return_values.append(f(parameters, closure))
            del parameters[-1]
        else:
            cursor_stack.append(-1)

    return return_values

if __name__ == '__main__':
    # This standalone block of code is only used for tests.
    config = yaml.load(open("config/behavior/default.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    ps = param_sweep_space(config['datagen']['param_sweep'])
    print(ps)

    def f(params, closure):
        print('_'.join([x[0][-1] + '_' + str(x[1]) for x in params]))
        
    parameter_sweep(ps, f)


            