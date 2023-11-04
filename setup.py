import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ks_evaluation',
    version='1.0.0',
    author='Luís Guilherme Ribeiro',
    author_email='luisguicomp@gmail.com',
    description='Library for calculating the KS of a predictive model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/luisguicomp/ks_evaluation',
    project_urls = {
        "Bug Tracker": "https://github.com/luisguicomp/ks_evaluation/issues"
    },
    license='LGR - Data Science',
    packages=['ks_evaluation'],
    install_requires=['pandas', 'scikit-learn', 'matplotlib']
)
