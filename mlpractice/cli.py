import argparse
from utils import init


def command_line():
    """Parse the user's input and execute the specified command."""
    command_functions = [
        init,
    ]

    command_to_function = {
        func.__name__: func for func in command_functions
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command',
        metavar='<command>',
        choices=command_to_function.keys(),
        help='The command to execute from the list: {0}'.format(
            list(
                command_to_function.keys(),
            ),
        ),
    )

    args = parser.parse_args()
    command_to_function[args.command]()
