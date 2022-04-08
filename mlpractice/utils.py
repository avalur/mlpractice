import inspect
import os
import re
import sys
from distutils.dir_util import copy_tree
from IPython import get_ipython

import mlpractice


class StopExecution(Exception):
    """A custom exception made for silent kernel interruption.
    Being raised outputs no message under the cell in Jupyter notebooks.
    """
    def _render_traceback_(self):
        """A special method that is responsible for rendering the exception
        in Jupyter notebooks.
        """
        pass


class ExceptionInterception:
    """A context manager that intercepts any exception coming from within it
    and immediately outputs the exception content. Thus, printed exception
    traceback doesn't include any information about the code beyond the
    context manager.

    Examples
    --------
    >>> with ExceptionInterception():
    ...     ...
    """
    def __init__(self):
        self.ip = get_ipython()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_val is not None:
            # print the exception message and traceback
            self.ip.showtraceback((exc_type, exc_val, traceback))
            # raise a silent exception to interrupt the kernel
            raise StopExecution
        return True


def get_source(match: re.Match) -> str:
    """Extract source code lines of a matched object and prepare them
    to being injected into .ipynb code cell."""
    match_object = eval(match.group()[9:-1])
    source_lines = inspect.getsourcelines(match_object)[0]
    new_lines = [line.rstrip() for line in source_lines]

    escape_symbols_re = re.compile(r'([\\"])')

    new_lines = [
        r'"{0}\n",'.format(escape_symbols_re.sub(r'\\\1', line))
        for line
        in new_lines
    ]

    # delete the redundant `"` prefix of the first line
    new_lines[0] = new_lines[0][1:]
    # delete the redundant `\n",` suffix of the last line
    new_lines[-1] = new_lines[-1][:-4]

    return ''.join(new_lines)


def inject_sources_into_template(file_path: str):
    """Inject python source code into the code cells of .ipynb file
    in places marked with
    #!source<python_object>
    """
    with open(file_path, 'rt') as target_file:
        file_as_text = target_file.read()

    # compile a regular expression that finds the #!source<python_object> marks
    reg_exp = re.compile(r'#!source<.+?>')
    # find all the occurrences of `reg_exp` in `file_as_text`
    # and replace them using the `get_source` function
    modified_file_as_text = reg_exp.sub(get_source, file_as_text)

    with open(file_path, 'wt') as target_file:
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
