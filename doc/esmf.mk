# ESMF application makefile fragment
#
# Use the following ESMF_ variables to compile and link
# your ESMF application against this ESMF build.
#
# !!! VERY IMPORTANT: If the location of this ESMF build is   !!!
# !!! changed, e.g. libesmf.a is copied to another directory, !!!
# !!! this file - esmf.mk - must be edited to adjust to the   !!!
# !!! correct new path                                        !!!
#
# Please see end of file for options used on this ESMF build
#

#----------------------------------------------
ESMF_VERSION_STRING=8.4.1
# Not a Git repository
ESMF_VERSION_STRING_GIT=NoGit
#----------------------------------------------

ESMF_VERSION_MAJOR=8
ESMF_VERSION_MINOR=4
ESMF_VERSION_REVISION=1
ESMF_VERSION_PATCHLEVEL=1
ESMF_VERSION_PUBLIC='T'
ESMF_VERSION_BETASNAPSHOT='F'


ESMF_APPSDIR=/home/valeriu/miniconda3/envs/py11/bin
ESMF_LIBSDIR=/home/valeriu/miniconda3/envs/py11/lib
ESMF_ESMXDIR=/home/valeriu/miniconda3/envs/py11/include/ESMX


ESMF_F90COMPILER=mpif90
ESMF_F90LINKER=mpif90

ESMF_F90COMPILEOPTS=-fallow-argument-mismatch -O -fPIC -m64 -mcmodel=small -pthread -ffree-line-length-none -fopenmp
ESMF_F90COMPILEPATHS=-I/home/valeriu/miniconda3/envs/py11/mod -I/home/valeriu/miniconda3/envs/py11/include -I/home/valeriu/miniconda3/envs/py11/include -I/home/valeriu/miniconda3/envs/py11/include
ESMF_F90COMPILECPPFLAGS=-DESMF_NO_INTEGER_1_BYTE -DESMF_NO_INTEGER_2_BYTE -DESMF_MOAB=1 -DESMF_LAPACK=1 -DESMF_LAPACK_INTERNAL=1 -DESMF_NO_ACC_SOFTWARE_STACK=1 -DESMF_NETCDF=1 -DESMF_YAMLCPP=1 -DESMF_YAML=1 -DESMF_PIO=1 -DESMF_NO_OPENACC -DESMF_BOPT_O -DESMF_TESTCOMPTUNNEL -DSx86_64_small=1 -DESMF_OS_Linux=1 -DESMF_COMM=mpich -DESMF_DIR=/home/conda/feedstock_root/build_artifacts/esmf_1679088787877/work
ESMF_F90COMPILEFREECPP=
ESMF_F90COMPILEFREENOCPP=-ffree-form
ESMF_F90COMPILEFIXCPP=-cpp -ffixed-form
ESMF_F90COMPILEFIXNOCPP=

ESMF_F90LINKOPTS= -m64 -mcmodel=small -pthread -Wl,--no-as-needed -fopenmp
ESMF_F90LINKPATHS=-L/home/valeriu/miniconda3/envs/py11/lib -L/home/valeriu/miniconda3/envs/py11/lib -L/home/valeriu/miniconda3/envs/py11/lib -L/home/valeriu/miniconda3/envs/py11/bin/../lib/gcc/x86_64-conda-linux-gnu/11.3.0/
ESMF_F90ESMFLINKPATHS=-L/home/valeriu/miniconda3/envs/py11/lib
ESMF_F90LINKRPATHS=-Wl,-rpath,/home/valeriu/miniconda3/envs/py11/lib -Wl,-rpath,/home/valeriu/miniconda3/envs/py11/lib  -Wl,-rpath,/home/valeriu/miniconda3/envs/py11/lib -Wl,-rpath,/home/valeriu/miniconda3/envs/py11/bin/../lib/gcc/x86_64-conda-linux-gnu/11.3.0/
ESMF_F90ESMFLINKRPATHS=-Wl,-rpath,/home/valeriu/miniconda3/envs/py11/lib
ESMF_F90LINKLIBS=-lrt -lstdc++ -ldl -lnetcdff -lnetcdf -lpioc
ESMF_F90ESMFLINKLIBS=-lesmf -lrt -lstdc++ -ldl -lnetcdff -lnetcdf -lpioc

ESMF_CXXCOMPILER=mpicxx
ESMF_CXXLINKER=mpicxx

ESMF_CXXCOMPILEOPTS=-std=c++11 -O -DNDEBUG -fPIC -DESMF_LOWERCASE_SINGLEUNDERSCORE -m64 -mcmodel=small -pthread -fopenmp
ESMF_CXXCOMPILEPATHS=-I/home/valeriu/miniconda3/envs/py11/include  -I/home/valeriu/miniconda3/envs/py11/include -I/home/valeriu/miniconda3/envs/py11/include -I/home/conda/feedstock_root/build_artifacts/esmf_1679088787877/work/src/prologue/yaml-cpp/include
ESMF_CXXCOMPILECPPFLAGS=-DESMF_NO_INTEGER_1_BYTE -DESMF_NO_INTEGER_2_BYTE -DESMF_MOAB=1 -DESMF_LAPACK=1 -DESMF_LAPACK_INTERNAL=1 -DESMF_NO_ACC_SOFTWARE_STACK=1 -DESMF_NETCDF=1 -DESMF_YAMLCPP=1 -DESMF_YAML=1 -DESMF_PIO=1 -DESMF_NO_OPENACC -DESMF_BOPT_O -DESMF_TESTCOMPTUNNEL -DSx86_64_small=1 -DESMF_OS_Linux=1 -DESMF_COMM=mpich -DESMF_DIR=/home/conda/feedstock_root/build_artifacts/esmf_1679088787877/work -D__SDIR__='' -DESMF_CXXSTD=11

ESMF_CXXLINKOPTS= -m64 -mcmodel=small -pthread -Wl,--no-as-needed -fopenmp
ESMF_CXXLINKPATHS=-L/home/valeriu/miniconda3/envs/py11/lib -L/home/valeriu/miniconda3/envs/py11/lib -L/home/valeriu/miniconda3/envs/py11/lib -L/home/valeriu/miniconda3/envs/py11/bin/../lib/gcc/x86_64-conda-linux-gnu/11.3.0/../../../../x86_64-conda-linux-gnu/lib/../lib/
ESMF_CXXESMFLINKPATHS=-L/home/valeriu/miniconda3/envs/py11/lib
ESMF_CXXLINKRPATHS=-Wl,-rpath,/home/valeriu/miniconda3/envs/py11/lib -Wl,-rpath,/home/valeriu/miniconda3/envs/py11/lib  -Wl,-rpath,/home/valeriu/miniconda3/envs/py11/lib -Wl,-rpath,/home/valeriu/miniconda3/envs/py11/bin/../lib/gcc/x86_64-conda-linux-gnu/11.3.0/../../../../x86_64-conda-linux-gnu/lib/../lib/
ESMF_CXXESMFLINKRPATHS=-Wl,-rpath,/home/valeriu/miniconda3/envs/py11/lib
ESMF_CXXLINKLIBS=-lrt -lgfortran -ldl -lnetcdff -lnetcdf -lpioc
ESMF_CXXESMFLINKLIBS=-lesmf -lrt -lgfortran -ldl -lnetcdff -lnetcdf -lpioc

ESMF_SO_F90COMPILEOPTS=-fPIC
ESMF_SO_F90LINKOPTS=-shared
ESMF_SO_F90LINKOPTSEXE=-Wl,-export-dynamic
ESMF_SO_CXXCOMPILEOPTS=-fPIC
ESMF_SO_CXXLINKOPTS=-shared
ESMF_SO_CXXLINKOPTSEXE=-Wl,-export-dynamic

ESMF_OPENMP_F90COMPILEOPTS=-fopenmp
ESMF_OPENMP_F90LINKOPTS=-fopenmp
ESMF_OPENMP_CXXCOMPILEOPTS=-fopenmp
ESMF_OPENMP_CXXLINKOPTS=-fopenmp

ESMF_OPENACC_F90COMPILEOPTS=-fopenacc
ESMF_OPENACC_F90LINKOPTS=-fopenacc
ESMF_OPENACC_CXXCOMPILEOPTS=-fopenacc
ESMF_OPENACC_CXXLINKOPTS=-fopenacc

# ESMF Tracing compile/link options
ESMF_TRACE_LDPRELOAD=/home/valeriu/miniconda3/envs/py11/lib/libesmftrace_preload.so
ESMF_TRACE_STATICLINKOPTS=-static -Wl,--wrap=c_esmftrace_notify_wrappers -Wl,--wrap=c_esmftrace_isinitialized -Wl,--wrap=write -Wl,--wrap=writev -Wl,--wrap=pwrite -Wl,--wrap=read -Wl,--wrap=open -Wl,--wrap=MPI_Allgather -Wl,--wrap=MPI_Allgatherv -Wl,--wrap=MPI_Allreduce -Wl,--wrap=MPI_Alltoall -Wl,--wrap=MPI_Alltoallv -Wl,--wrap=MPI_Alltoallw -Wl,--wrap=MPI_Barrier -Wl,--wrap=MPI_Bcast -Wl,--wrap=MPI_Bsend -Wl,--wrap=MPI_Gather -Wl,--wrap=MPI_Gatherv -Wl,--wrap=MPI_Iprobe -Wl,--wrap=MPI_Irecv -Wl,--wrap=MPI_Irsend -Wl,--wrap=MPI_Isend -Wl,--wrap=MPI_Issend -Wl,--wrap=MPI_Probe -Wl,--wrap=MPI_Recv -Wl,--wrap=MPI_Reduce -Wl,--wrap=MPI_Rsend -Wl,--wrap=MPI_Scan -Wl,--wrap=MPI_Scatter -Wl,--wrap=MPI_Scatterv -Wl,--wrap=MPI_Send -Wl,--wrap=MPI_Sendrecv -Wl,--wrap=MPI_Test -Wl,--wrap=MPI_Testall -Wl,--wrap=MPI_Testany -Wl,--wrap=MPI_Testsome -Wl,--wrap=MPI_Wait -Wl,--wrap=MPI_Waitall -Wl,--wrap=MPI_Waitany -Wl,--wrap=MPI_Waitsome -Wl,--wrap=mpi_allgather_ -Wl,--wrap=mpi_allgather__ -Wl,--wrap=mpi_allgatherv_ -Wl,--wrap=mpi_allgatherv__ -Wl,--wrap=mpi_allreduce_ -Wl,--wrap=mpi_allreduce__ -Wl,--wrap=mpi_alltoall_ -Wl,--wrap=mpi_alltoall__ -Wl,--wrap=mpi_alltoallv_ -Wl,--wrap=mpi_alltoallv__ -Wl,--wrap=mpi_alltoallw_ -Wl,--wrap=mpi_alltoallw__ -Wl,--wrap=mpi_barrier_ -Wl,--wrap=mpi_barrier__ -Wl,--wrap=mpi_bcast_ -Wl,--wrap=mpi_bcast__ -Wl,--wrap=mpi_bsend_ -Wl,--wrap=mpi_bsend__ -Wl,--wrap=mpi_exscan_ -Wl,--wrap=mpi_exscan__ -Wl,--wrap=mpi_gather_ -Wl,--wrap=mpi_gather__ -Wl,--wrap=mpi_gatherv_ -Wl,--wrap=mpi_gatherv__ -Wl,--wrap=mpi_iprobe_ -Wl,--wrap=mpi_iprobe__ -Wl,--wrap=mpi_irecv_ -Wl,--wrap=mpi_irecv__ -Wl,--wrap=mpi_irsend_ -Wl,--wrap=mpi_irsend__ -Wl,--wrap=mpi_isend_ -Wl,--wrap=mpi_isend__ -Wl,--wrap=mpi_issend_ -Wl,--wrap=mpi_issend__ -Wl,--wrap=mpi_probe_ -Wl,--wrap=mpi_probe__ -Wl,--wrap=mpi_recv_ -Wl,--wrap=mpi_recv__ -Wl,--wrap=mpi_reduce_ -Wl,--wrap=mpi_reduce__ -Wl,--wrap=mpi_reduce_scatter_ -Wl,--wrap=mpi_reduce_scatter__ -Wl,--wrap=mpi_rsend_ -Wl,--wrap=mpi_rsend__ -Wl,--wrap=mpi_scatter_ -Wl,--wrap=mpi_scatter__ -Wl,--wrap=mpi_scatterv_ -Wl,--wrap=mpi_scatterv__ -Wl,--wrap=mpi_scan_ -Wl,--wrap=mpi_scan__ -Wl,--wrap=mpi_send_ -Wl,--wrap=mpi_send__ -Wl,--wrap=mpi_sendrecv_ -Wl,--wrap=mpi_sendrecv__ -Wl,--wrap=mpi_test_ -Wl,--wrap=mpi_test__ -Wl,--wrap=mpi_testall_ -Wl,--wrap=mpi_testall__ -Wl,--wrap=mpi_testany_ -Wl,--wrap=mpi_testany__ -Wl,--wrap=mpi_testsome_ -Wl,--wrap=mpi_testsome__ -Wl,--wrap=mpi_wait_ -Wl,--wrap=mpi_wait__ -Wl,--wrap=mpi_waitall_ -Wl,--wrap=mpi_waitall__ -Wl,--wrap=mpi_waitany_ -Wl,--wrap=mpi_waitany__
ESMF_TRACE_STATICLINKLIBS=-lesmftrace_static

# Internal ESMF variables, do NOT depend on these!

ESMF_INTERNAL_DIR=/home/conda/feedstock_root/build_artifacts/esmf_1679088787877/work
ESMF_INTERNAL_MPIRUN="mpirun "

#
# !!! The following options were used on this ESMF build !!!
#
# ESMF_DIR: /home/conda/feedstock_root/build_artifacts/esmf_1679088787877/work
# ESMF_OS: Linux
# ESMF_MACHINE: x86_64
# ESMF_ABI: 64
# ESMF_COMPILER: gfortran
# ESMF_BOPT: O
# ESMF_COMM: mpich
# ESMF_SITE: default
# ESMF_PTHREADS: ON
# ESMF_OPENMP: ON
# ESMF_OPENACC: OFF
# ESMF_ARRAY_LITE: FALSE
# ESMF_NO_INTEGER_1_BYTE: TRUE
# ESMF_NO_INTEGER_2_BYTE: TRUE
# ESMF_FORTRANSYMBOLS: default
# ESMF_MAPPER_BUILD: OFF
# ESMF_AUTO_LIB_BUILD: ON
# ESMF_DEFER_LIB_BUILD: ON
# ESMF_SHARED_LIB_BUILD: ON
# 
# ESMF environment variables pointing to 3rd party software:
# ESMF_MOAB:              internal
# ESMF_LAPACK:            internal
# ESMF_ACC_SOFTWARE_STACK:            none
# ESMF_NETCDF:            split
# ESMF_NETCDF_INCLUDE:    /home/valeriu/miniconda3/envs/py11/include
# ESMF_NETCDF_LIBS:       -lnetcdff -lnetcdf
# ESMF_NETCDF_LIBPATH:    /home/valeriu/miniconda3/envs/py11/lib
# ESMF_PIO:               external
# ESMF_PIO_INCLUDE:       /home/valeriu/miniconda3/envs/py11/include
# ESMF_PIO_LIBS:          -lpioc
# ESMF_PIO_LIBPATH:       /home/valeriu/miniconda3/envs/py11/lib
# ESMF_YAMLCPP:           internal
#
# * Compilers, Linkers, Flags, and Libraries *
# Location of the preprocessor:      /home/valeriu/miniconda3/envs/py11/bin/x86_64-conda-linux-gnu-cpp
# Location of the Fortran compiler:  /home/valeriu/miniconda3/envs/py11/bin/mpif90
# Location of the Fortran linker:    /home/valeriu/miniconda3/envs/py11/bin/mpif90
# Location of the C++ compiler:      /home/valeriu/miniconda3/envs/py11/bin/mpicxx
# Location of the C++ linker:        /home/valeriu/miniconda3/envs/py11/bin/mpicxx
# Location of the C compiler:        /home/valeriu/miniconda3/envs/py11/bin/mpicc
# Location of the C linker:          /home/valeriu/miniconda3/envs/py11/bin/mpicc
#
# !!! ----- User set ESMF_ environment variables ----- !!!
#
#  --------------------------------------------------------------
#   * User set ESMF environment variables *
#  ESMF_COMM=mpich3
#  ESMF_CPP=/home/valeriu/miniconda3/envs/py11/bin/x86_64-conda-linux-gnu-cpp -E -P -x c
#  ESMF_CXX=mpicxx
#  ESMF_DIR=/home/conda/feedstock_root/build_artifacts/esmf_1679088787877/work
#  ESMF_F90COMPILEOPTS=-fallow-argument-mismatch
#  ESMF_F90=mpif90
#  ESMF_INSTALL_BINDIR=/home/valeriu/miniconda3/envs/py11/bin
#  ESMF_INSTALL_DOCDIR=/home/valeriu/miniconda3/envs/py11/doc
#  ESMF_INSTALL_HEADERDIR=/home/valeriu/miniconda3/envs/py11/include
#  ESMF_INSTALL_LIBDIR=/home/valeriu/miniconda3/envs/py11/lib
#  ESMF_INSTALL_MODDIR=/home/valeriu/miniconda3/envs/py11/mod
#  ESMF_INSTALL_PREFIX=/home/valeriu/miniconda3/envs/py11
#  ESMF_NETCDF_INCLUDE=/home/valeriu/miniconda3/envs/py11/include
#  ESMF_NETCDF_LIBPATH=/home/valeriu/miniconda3/envs/py11/lib
#  ESMF_NETCDF=split
#  ESMF_PIO=external
#  ESMF_PIO_INCLUDE=/home/valeriu/miniconda3/envs/py11/include
#  ESMF_PIO_LIBPATH=/home/valeriu/miniconda3/envs/py11/lib
#  
#  --------------------------------------------------------------
