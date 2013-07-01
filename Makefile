all: pyqpbo

pyqpbo: 
	python setup.py build_ext --inplace
