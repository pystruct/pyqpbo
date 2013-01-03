all: pyqpbo

pyqpbo: qpbo_src
	python setup.py build_ext --inplace

QPBO-v1.3.src.tar.gz:
	wget http://pub.ist.ac.at/~vnk/software/QPBO-v1.3.src.tar.gz

qpbo_src: QPBO-v1.3.src.tar.gz
	tar -xvf QPBO-v1.3.src.tar.gz
	mv QPBO-v1.3.src qpbo_src
