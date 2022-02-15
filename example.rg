import "regent"
local c = regentlib.c

terralib.linklibrary("/home/seshu/pkg/lib/libcblas.so")
local cblas = terralib.includec("openblas/cblas.h", {"-I", "/home/seshu/pkg/include"})

-- terralib.linklibrary("/scratch2/seshu/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so")
-- terralib.linklibrary("/scratch2/seshu/intel/oneapi/mkl/2022.0.1/lib/intel64/libmkl_rt.so")
-- local cblas = terralib.includec("mkl_cblas.h", {"-I", "/scratch2/seshu/intel/oneapi/mkl/latest/include"})

local blas = terralib.includec("blas_tasks.h", {"-I", "./build"})
terralib.linklibrary("./build/blas_tasks.so")

terralib.linklibrary("./build/libcontext_manager.so")

require("blas")


task main()
  --- Make sure to use single threaded blas
  -- cblas.openblas_set_num_threads(1)

  var A = region(ispace(int2d, {2, 2}), double)
  var B = region(ispace(int2d, {2, 2}), double)
  var C = region(ispace(int2d, {2, 2}), double)

  fill(A, 0)
  fill(B, 0)
  fill(C, 0)
  A[{0, 0}] = 1
  A[{1, 0}] = 1
  A[{1, 1}] = 1

  B[{0, 1}] = 1.5
  B[{1, 0}] = 0.5

  dgemm(cblas.CblasColMajor, cblas.CblasNoTrans, cblas.CblasNoTrans, 1, A, B, 2, C)

  for i in C.ispace do
    c.printf("C[%d, %d] = %0.3f\n", i.x, i.y, C[i])
  end

end

regentlib.start(main, blas.blas_tasks_h_register)


