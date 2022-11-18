import sys
from os import environ as env
from setuptools import setup


if __name__ == '__main__':
    try:
        with open('.env', 'r') as fh:
            env_dict = dict(tuple(line.replace("\n","").split('=')) 
                for line in fh.readlines() if not line.startswith('#'))
            env.update(env_dict)
    except Exception as e:
        raise e
    # avoid creation of __pycache__ folders
    sys.dont_write_bytecode = True
    setup(version=env["VERSION"])
