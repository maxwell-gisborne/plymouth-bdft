import pytest

from plymouth_bdft.condlib import version as condlib_version

def pytest_report_header(config):
    return  f'condlib version: {condlib_version()}'


