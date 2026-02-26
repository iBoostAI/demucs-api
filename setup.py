# Demucs v4 Inference-only API version
# Inference-only fork: https://github.com/cj-mills/cjm-demucs-v4
# Original: # https://github.com/adefossez/demucs
# License: MIT

from pathlib import Path

from setuptools import setup

NAME = 'cjm-demucs-v4'
DESCRIPTION = 'Inference-only fork of Demucs v4 that provides audio source separation with TorchCodec replacing torchaudio I/O.'

URL = 'https://github.com/iBoostAI/demucs-api'
AUTHOR = 'iBoostAI'
REQUIRES_PYTHON = '>=3.12'

HERE = Path(__file__).parent

for line in open('demucs/__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']


def load_requirements(name):
    required = [i.strip() for i in open(HERE / name)]
    required = [i for i in required if not i.startswith('#') and i]
    return required


REQUIRED = load_requirements('requirements.txt')

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['demucs'],
    install_requires=REQUIRED,
    include_package_data=True,
    package_data={'demucs': ['remote/*.txt', 'remote/*.yaml']},
    entry_points={
        'console_scripts': ['demucs=demucs.separate:main'],
    },
    license='MIT License',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
    ],
)
