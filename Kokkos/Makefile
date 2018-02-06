EXE=vecadd-kokkos

include $(KOKKOS_PATH)/Makefile.kokkos

CXX = $(NVCC_WRAPPER)
CXXFLAGS= -Wall -O3
LIBS = -lm

all: $(EXE)

$(EXE): $(EXE).cpp
	$(CXX) $(CXXFLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $^ $(LIBS) $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) -o $@


clean:
	rm $(EXE)
