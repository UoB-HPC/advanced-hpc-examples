EXE=vecadd-opencl

CC=gcc
CFLAGS= -std=c99 -Wall -O3
LIBS = -lm

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS += -framework OpenCL
else
	LIBS += -lOpenCL
endif

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

clean:
	rm $(EXE)
