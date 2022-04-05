import inspect
import os
import re
import sys
from distutils.dir_util import copy_tree
from IPython import get_ipython

import mlpractice


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


class ExceptionInterception:
    def __init__(self):
        self.ip = get_ipython()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_val is not None:
            self.ip.showtraceback((exc_type, exc_val, traceback))
            raise StopExecution
        return True


def get_source(match):
    match_object = eval(match.group()[9:-1])
    source_lines = inspect.getsourcelines(match_object)[0]
    new_lines = [line.rstrip() for line in source_lines]

    escape_symbols_re = re.compile(r'([\\"])')

    new_lines = [
        r'"{0}\n",'.format(escape_symbols_re.sub(r'\\\1', line))
        for line
        in new_lines
    ]

    new_lines[0] = new_lines[0][1:]
    new_lines[-1] = new_lines[-1][:-4]

    return ''.join(new_lines)


def inject_sources_into_template(file_path):
    """Inject python source code into the file in places marked with
    #!source<python_object>
    """
    with open(file_path, 'r') as target_file:
        file_as_text = target_file.read()

    reg_exp = re.compile(r'#!source<.+?>')
    modified_file_as_text = reg_exp.sub(get_source, file_as_text)

    with open(file_path, 'w') as target_file:
        target_file.write(modified_file_as_text)


def init():
    """Initialize a directory with tasks."""
    if os.path.isdir('tasks'):
        print('Directory "tasks" already exists!')
        sys.exit(0)

    os.mkdir('tasks')
    os.chdir('tasks')

    templates_dir = os.path.join(
        os.path.dirname(inspect.getfile(mlpractice)),
        'templates',
    )
    tasks_dir = os.getcwd()

    copy_tree(templates_dir, tasks_dir)
    for dir_path, dir_names, filenames in os.walk(tasks_dir):
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            if file_path.endswith('.ipynb'):
                inject_sources_into_template(file_path)

    print(f'Initialized a directory with tasks at {tasks_dir}')
