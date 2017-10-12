# Makefile
# Generic Makefile for making cuda programs
#
BINDIR					:= ../bin
BIN					:= $(BINDIR)/mrg-cg2
# flags
CUDA_INSTALL_PATH	:= /usr/local/cuda
OBJDIR				:= $(BINDIR)/obj
INCLUDES			+= -I$(CUDA_INSTALL_PATH)/include -I.
LIBS				:= -L$(CUDA_INSTALL_PATH)/lib64
CFLAGS				:= -O3
LDFLAGS				:= -lrt -lm -lcudart -lcufft
# compilers
#NVCC				:= $(CUDA_INSTALL_PATH)/bin/nvcc --compiler-options -fpermissive -arch sm_20 --ptxas-options=-v -use_fast_math 
NVCC				:= $(CUDA_INSTALL_PATH)/bin/nvcc --compiler-options -fpermissive --ptxas-options=-v -use_fast_math
#NVCC				:= /usr/bin/nvcc --compiler-options -fpermissive --ptxas-options=-v -use_fast_math
#NVCC				:= $(CUDA_INSTALL_PATH)/bin/nvcc --compiler-options -fpermissive -arch sm_20 --ptxas-options=-v
CC					:= g++
LINKER				:= g++ -fPIC
# files
CPP_SOURCES			:= \
  main.cpp \
  ComputationalArrays.cpp \
  IO/paramio.cpp \
  IO/configreader.cpp \
  IO/dcdio.cpp \
  IO/psfio.cpp \
  IO/pdbio.cpp \
  IO/xyzio.cpp \
  IO/topio.cpp \
  Util/wrapper.cpp \
  Util/timer.cpp \
  Util/mystl.cpp
CU_SOURCES			:= \
  md.cu
  # Util/ReductionAlgorithms.cu \
  # Integrators/LeapFrogLangevin.cu
SOP_TOP_SOURCES		:= \
  TopologyCreators/SOP/sop-top2.cpp \
  TopologyCreators/SOP/Util/aatocg_pol.cpp \
  IO/configreader.cpp \
  IO/psfio.cpp \
  IO/pdbio.cpp \
  IO/xyzio.cpp \
  IO/topio.cpp \
  Util/wrapper.cpp \
  Util/mystl.cpp
CPP_OBJS				:= $(patsubst %.cpp, $(OBJDIR)/%.cpp.o, $(CPP_SOURCES))
CU_OBJS				:= $(patsubst %.cu, $(OBJDIR)/%.cu.o, $(CU_SOURCES))
 
$(BIN): clean makedirs $(CU_OBJS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(CPP_SOURCES) $(LDFLAGS) $(INCLUDES) $(LIBS)
	$(CC)  -o $(BINDIR)/sop-top2 $(SOP_TOP_SOURCES)
 
$(OBJDIR)/%.c.o: $(CPP_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<
 
$(OBJDIR)/%.cu.o: $(CU_SOURCES)
	$(NVCC) $(INCLUDES) -o $@ -c $<
 
makedirs:
	mkdir -p $(BINDIR)
	mkdir -p $(OBJDIR)

run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_INSTALL_PATH)/lib ./$(BIN)
 
clean:
	rm -rf $(BINDIR)
	
install:
	cp $(BIN) /usr/bin/$(BIN)
	
