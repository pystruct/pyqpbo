all: pyqpbo

pyqpbo: libqpbo.so
	python setup.py build_ext --inplace

QPBO-v1.3.src.tar.gz:
	wget http://pub.ist.ac.at/~vnk/software/QPBO-v1.3.src.tar.gz

qpbo_src: QPBO-v1.3.src.tar.gz
	tar -xvf QPBO-v1.3.src.tar.gz
	mv QPBO-v1.3.src qpbo_src

libqpbo.so: qpbo_src
	g++ -fPIC -shared -Iqpbo_src qpbo_src/QPBO.cpp  qpbo_src/QPBO_extra.cpp  qpbo_src/QPBO_maxflow.cpp  qpbo_src/QPBO_postprocessing.cpp  -o libqpbo.so
