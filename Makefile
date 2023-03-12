all: build install

build: clean makedir
	cd build && cmake .. && make

install:
	cd build && make install

.PHONY: clean makedir run
clean:
	rm -rf build

makedir:
	mkdir build

run:
	./install/bin/run

cmake_test: clean mkdir
	cd build && cmake ..