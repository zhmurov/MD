# Makefile
# Generic Makefile for making cuda programs
#
BIN					:= mrg-cg2
# flags
CUDA_INSTALL_PATH	:= /usr/local/cuda
OBJDIR				:= obj
INCLUDES			+= -I$(CUDA_INSTALL_PATH)/include -I.
LIBS				:= -L$(CUDA_INSTALL_PATH)/lib64
CFLAGS				:= -O3
LDFLAGS				:= -lrt -lm -lcudart -lcufft
# compilers
#NVCC				:= $(CUDA_INSTALL_PATH)/bin/nvcc --compiler-options -fpermissive -arch sm_20 --ptxas-options=-v -use_fast_math 
NVCC				:= $(CUDA_INSTALL_PATH)/bin/nvcc --compiler-options -fpermissive -arch sm_20 --ptxas-options=-v -use_fast_math
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
CPP_OBJS				:= $(patsubst %.cpp, $(OBJDIR)/%.cpp.o, $(CPP_SOURCES))
CU_OBJS				:= $(patsubst %.cu, $(OBJDIR)/%.cu.o, $(CU_SOURCES))
 
$(BIN): makedirs clean $(CU_OBJS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(CPP_SOURCES) $(LDFLAGS) $(INCLUDES) $(LIBS)
 
$(OBJDIR)/%.c.o: $(CPP_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<
 
$(OBJDIR)/%.cu.o: $(CU_SOURCES)
	$(NVCC) $(INCLUDES) -o $@ -c $<
 
makedirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(OBJDIR)/Common
	mkdir -p $(OBJDIR)/Potentials
	mkdir -p $(OBJDIR)/Updaters
	mkdir -p $(OBJDIR)/Integrators

run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_INSTALL_PATH)/lib ./$(BIN)
 
clean:
	rm -f $(BIN) $(OBJDIR)/*.o
	
install:
	cp $(BIN) /usr/bin/$(BIN)
	
