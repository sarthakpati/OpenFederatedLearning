# Copyright (C) 2020 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('../'))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import sphinx_rtd_theme
import sphinxcontrib.napoleon

extensions = [
    "sphinx_rtd_theme",
    'sphinx.ext.autosectionlabel',
    "sphinxcontrib.napoleon",
    "rinoh.frontend.sphinx",
    "sphinx-prompt",
    'sphinx_substitution_extensions',
    "sphinx.ext.ifconfig",
    'sphinxcontrib.mermaid',
]

# -- Project information -----------------------------------------------------

# This will replace the |variables| within the rST documents automatically

#PRODUCT_VERSION = "Intel"
PRODUCT_VERSION = "OFL"

tags.add(PRODUCT_VERSION)

project = 'Open Federated Learning'
author = 'Intel Corporation'
master_doc = 'index'
version = "{}.{}".format(datetime.now().year, datetime.now().month)

# Global variables for rST
rst_prolog = """
.. |productName| replace:: Open Federated Learning
.. |productZip| replace:: OpenFederatedLearning.zip
.. |productDir| replace:: OpenFederatedLearning
.. |prod| replace:: OpenFL
.. |pkg| replace:: openfl
.. |pt| replace:: PyTorch
.. |tf| replace:: TensorFlow
.. |fets| replace:: FeTS-AI

.. _Makefile: https://github.com/IntelLabs/OpenFederatedLearning/blob/master/Makefile
"""

rinoh_documents = [('index', u'open_fl_manual', u'Open Federated Learning Manual', u'FeTS')]

napoleon_google_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "tutorials/*", "graveyard/*"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autosectionlabel_prefix_document = True
