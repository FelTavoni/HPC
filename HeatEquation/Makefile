EXEFILE   = mdf
CPUCC     = gcc
CPPFLAGS  = -std=c11 -m64 -O3
DEFS      =
INCLUDES  =
LIBDIR    =
LIBS     =  -lm
LINK     =  $(LIBDIR) $(LIBS)

CPU_COMPILE = $(CPUCC) $(DEFS) $(INCLUDES) $(CPPFLAGS)

all: seq cuda

seq:
	$(CPU_COMPILE) -c mdf.c
	$(CPU_COMPILE)	mdf.o $(LINK) -o $(EXEFILE) $(PAPILIBS)

cuda:
	nvcc mdf_cuda.cu -o mdf_cuda.exe
