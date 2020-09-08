import setuptools

setuptools.setup(
    name='thesis',
    version='0.1.0',
    author='Anton Mølbjerg Eskildsen',
    description='Code package used in my thesis',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'Click'
    ],
    entry_points='''
        [console_scripts]
        data=thesis.util.data_tool:data
        experiment=thesis.tools.filter_experiment:experiment
    '''
)