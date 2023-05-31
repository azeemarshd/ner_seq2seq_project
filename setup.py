from setuptools import setup

setup(
    name='ner_seq2seq_project',
    version='0.1.0',
    packages=['src', 'src.utils', 'notebooks'],
    install_requires=[
        'tensorflow',  # Example neural framework
        'numpy'
    ],
    author='Azeem Arshad',
    description='NER Project',
    url='https://github.com/azeemarshad97/ner_seq2seq_project',
)
