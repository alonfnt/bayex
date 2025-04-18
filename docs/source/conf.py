# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath('../../bayex'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'bayex'
copyright = '2025, Albert Alonso'
author = 'Albert Alonso'
release = '0.2.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'myst_parser',
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'sphinx.ext.autosummary',
        'sphinx_autodoc_typehints',
        'sphinx.ext.mathjax',
        ]

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'special-members': False,
    'inherited-members': True,
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ["_static"]
html_logo = "_static/bayex.png"
