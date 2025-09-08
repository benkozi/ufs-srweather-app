help([[
This module loads libraries for building the UFS SRW App on
the MSU machine Hercules using intel-oneapi-compilers/2022.2.1
]])

whatis([===[Loads libraries needed for building the UFS SRW App on Hercules ]===])

prepend_path("MODULEPATH", "/apps/contrib/spack-stack/spack-stack-1.9.2/envs/ue-oneapi-2024.1.0/install/modulefiles/Core")
prepend_path("MODULEPATH", "/apps/contrib/spack-stack/spack-stack-1.9.2/envs/ue-oneapi-2024.1.0/install/modulefiles/intel-oneapi-mpi/2021.13-sqiixt7/gcc/13.3.0")

load("stack-oneapi/2024.2.1")
load("stack-intel-oneapi-mpi/2021.13")
load("stack-python/3.11.7")
load("cmake/3.27.9")

load("srw_common")
load("zlib/1.2.13")

load("nccmp/1.9.0.1")
load("nco/5.2.4")
load(pathJoin("prod_util", os.getenv("prod_util_ver") or "2.1.1"))

setenv("CFLAGS","-diag-disable=10441")
setenv("FFLAGS","-diag-disable=10441")

setenv("FC", "mpiifort")

setenv("CMAKE_C_COMPILER","mpiicc")
setenv("CMAKE_CXX_COMPILER","mpiicpc")
setenv("CMAKE_Fortran_COMPILER","mpiifort")
setenv("CMAKE_Platform","hercules.intel")
