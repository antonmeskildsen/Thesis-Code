import setuptools

setuptools.setup(
    name='thesis',
    version='0.1.0',
    author='Anton Mølbjerg Eskildsen',
    description='Code package used in my thesis',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pupilfit',
        'scikit-image',
        'scipy',
        'matplotlib',
        'streamlit',
        'pandas',
        'seaborn',
        'Click'
    ],
    entry_points='''
        [console_scripts]
        data=thesis.tools.cli.data_tool:data
        filter-exp=thesis.tools.cli.filter_experiment:main
    '''
)