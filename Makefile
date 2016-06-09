# taken from sklearn and simplified.

PYTHON ?= python3
CYTHON ?= cython
NOSETESTS ?= nosetests3
CTAGS ?= ctags

all: clean inplace test

clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(NOSETESTS) -s -v uplift
test-sphinxext:
	$(NOSETESTS) -s -v doc/sphinxext/

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s -v --with-coverage uplift

test: test-code test-sphinxext test-doc

trailing-spaces:
	find uplift -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

cython:
	python build_tools/cythonize.py uplift

ctags:
	$(CTAGS) --python-kinds=-i -R sklearn

doc: inplace
	$(MAKE) -C doc html

doc-noplot: inplace
	$(MAKE) -C doc html-noplot

code-analysis:
	flake8 uplift | grep -v __init__ | grep -v external
	pylint -E -i y uplift/ -d E1103,E0611,E1101
