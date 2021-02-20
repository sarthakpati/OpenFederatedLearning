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

# Docker support removed from Makefile for now

python_version ?= python3

# FIXME: how can we use this to help detect unsupported python versions
python_minor_version := $(shell python3 -c 'import sys; print(sys.version_info.minor)')

# our phony targets
.PHONY: install_openfl
.PHONY: install_openfl_pytorch
.PHONY: install_openfl_tensorflow
.PHONY: install remove_build
.PHONY: uninstall
.PHONY: uninstall_openfl
.PHONY: uninstall_openfl_pytorch
.PHONY: uninstall_openfl_tensorflow
.PHONY: reinstall
.PHONY: clean
.PHONY: openfl_whl
.PHONY: openfl_pytorch_whl
.PHONY: openfl_tensorflow_whl
.PHONY: fets_whl
.PHONY: gandlf_whl

# FIXME: some real makefile fu could probably make this all easier

# the virtual environment
venv = venv/bin/$(python_version)

# used to determine if the packages are installed
openfl				= venv/lib/$(python_version)/site-packages/openfl
openfl_pytorch 		= venv/lib/$(python_version)/site-packages/openfl/models/pytorch
openfl_tensorflow 	= venv/lib/$(python_version)/site-packages/openfl/models/tensorflow
fets		 		= venv/lib/$(python_version)/site-packages/fets
gandlf		 		= venv/lib/$(python_version)/site-packages/GANDLF

# the wheel files for the packages
openfl_whl 				= dist/openfl-0.0.1-py3-none-any.whl
openfl_pytorch_whl 		= dist/openfl.pytorch-0.0.1-py3-none-any.whl
openfl_tensorflow_whl 	= dist/openfl.tensorflow-0.0.1-py3-none-any.whl
fets_whl				= submodules/fets_ai/Algorithms/dist/fets-0.0.1-py3-none-any.whl
gandlf_whl				= submodules/fets_ai/Algorithms/GANDLF/dist/GANDLF-0.0.7.dev0-py3-none-any.whl

# the python virtual env recipe
$(venv):
	$(python_version) -m venv venv
	# specifying pip version to avoid issue with later intel-tensorflow install
	venv/bin/pip3 install --force-reinstall pip==19.0
	venv/bin/pip3 install --upgrade setuptools
	venv/bin/pip3 install --upgrade wheel
	venv/bin/pip3 install --upgrade pyyaml

# the wheel recipes
$(openfl_whl): $(venv) remove_build
	rm -rf build
	rm -rf dist
	$(venv) setup.py bdist_wheel
	# we will use the wheel, and do not want the egg info
	rm -rf openfl.egg-info

$(openfl_pytorch_whl): $(venv) remove_build
	rm -rf build
	rm -rf dist
	$(venv) setup_pytorch.py bdist_wheel
	# we will use the wheel, and do not want the egg info
	rm -rf openfl.pytorch.egg-info

$(openfl_tensorflow_whl): $(venv) remove_build
	rm -rf build
	rm -rf dist
	$(venv) setup_tensorflow.py bdist_wheel
	# we will use the wheel, and do not want the egg info
	rm -rf openfl.tensorflow.egg-info

$(fets_whl): $(venv)
	cd submodules/fets_ai/Algorithms && rm -rf build
	cd submodules/fets_ai/Algorithms && rm -rf dist
	cd submodules/fets_ai/Algorithms && ../../../$(venv) setup.py bdist_wheel
	# we will use the wheel, and do not want the egg info
	cd submodules/fets_ai/Algorithms && rm -rf fets.egg-info

$(gandlf_whl): $(venv)
	cd submodules/fets_ai/Algorithms/GANDLF && rm -rf build
	cd submodules/fets_ai/Algorithms/GANDLF && rm -rf dist
	cd submodules/fets_ai/Algorithms/GANDLF && ../../../../$(venv) setup.py bdist_wheel
	# we will use the wheel, and do not want the egg info
	cd submodules/fets_ai/Algorithms/GANDLF && rm -rf GANDLF.egg-info

# the install recipes
$(openfl): $(openfl_whl)
	venv/bin/pip install $(openfl_whl)

$(openfl_pytorch): $(openfl_pytorch_whl)
	venv/bin/pip install $(openfl_pytorch_whl)

$(openfl_tensorflow): $(openfl_tensorflow_whl) 
	venv/bin/pip install $(openfl_tensorflow_whl)

$(fets): $(gandlf_whl) $(fets_whl)
	venv/bin/pip install $(fets_whl)
	venv/bin/pip install $(gandlf_whl) 
	

install_openfl: $(openfl)

install_openfl_pytorch: $(openfl_pytorch) $(openfl)

install_openfl_tensorflow: $(openfl_tensorflow) $(openfl)

install_fets: $(fets) $(openfl_pytorch) $(openfl)

install: install_openfl install_openfl_pytorch install_openfl_tensorflow

# the uninstall recipes
remove_build:
	rm -rf build
	rm -rf dist

uninstall: uninstall_openfl uninstall_openfl_pytorch uninstall_openfl_tensorflow uninstall_fets

uninstall_openfl: remove_build
	venv/bin/pip uninstall -y openfl

uninstall_openfl_pytorch: remove_build
	venv/bin/pip uninstall -y openfl.pytorch

uninstall_openfl_tensorflow: remove_build
	venv/bin/pip uninstall -y openfl.tensorflow
	
uninstall_fets:
	rm -rf submodules/fets_ai/Algorithms/build
	rm -rf submodules/fets_ai/Algorithms/dist
	rm -rf submodules/fets_ai/Algorithms/GANDLF/build
	rm -rf submodules/fets_ai/Algorithms/GANDLF/dist
	venv/bin/pip uninstall -y fets
	venv/bin/pip uninstall -y GANDLF

# whl file recipes

openfl_whl				: $(openfl_whl)
openfl_pytorch_whl		: $(openfl_pytorch_whl)
openfl_tensorflow_whl	: $(openfl_tensorflow_whl)
fets_whl				: $(fets_whl)
gandlf_whl				: $(gandlf_whl)

# the reinstall recipe does everything by default
reinstall					: uninstall 					install
reinstall_openfl			: uninstall_openfl 				install_openfl
reinstall_openfl_tensorflow	: uninstall_openfl_tensorflow 	install_openfl_tensorflow
reinstall_openfl_pytorch	: uninstall_openfl_pytorch 		install_openfl_pytorch
reinstall_fets				: uninstall_fets 				install_fets

# blows up the venv
clean:
	rm -r -f venv
	rm -r -f dist
	rm -r -f build
	rm -rf submodules/fets_ai/Algorithms/build
	rm -rf submodules/fets_ai/Algorithms/dist
	rm -rf submodules/fets_ai/Algorithms/GANDLF/build
	rm -rf submodules/fets_ai/Algorithms/GANDLF/dist
