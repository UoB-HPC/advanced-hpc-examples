EXE = vecadd-kokkos
SRC = *.cpp
OBJ = $(notdir $(SRC:.cpp=.o))

all: $(EXE)
include $(KOKKOS_PATH)/Makefile.kokkos

CXX = $(NVCC_WRAPPER)
CXXFLAGS = -Wall -O3
LINK = $(CXX)
LIBS = -lm

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

clean:
	rm $(EXE) $(OBJ)
