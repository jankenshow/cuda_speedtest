all: build install

build: clean makedir
	cd build && cmake .. && make

install:
	cd build && make install

.PHONY: clean makedir
clean:
	rm -rf build

makedir:
	mkdir build

cmake_test: clean mkdir
	cd build && cmake ..