TARGET=main
OBJECTS=util.o matmul.o main.o

CPPFLAGS=-std=c++11 -O3 -Wall -march=native -mavx2 -mno-avx512f -mfma -fopenmp
CPPFLAGS+= -I/usr/local/cuda/include/
LDFLAGS=-lm -lcudart -lcublas -lnvToolsExt
LDFLAGS+=-L/usr/local/cuda/lib64

NVCC=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	g++ $(CPPFLAGS) $^ -o $@ $(LDFLAGS)

matmul.o: matmul.cu
	$(NVCC) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
