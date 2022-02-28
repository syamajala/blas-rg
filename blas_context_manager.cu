#include "blas_context_manager.h"

cublasHandle_t get_blas_handle()
{
  static __thread cublasHandle_t handle;

  if(handle == NULL)
  {
    cublasStatus_t stat;
    stat = cublasCreate(&handle);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      printf("CUBLAS initialization failed! Status: %d\n", stat);
    }
  }
  return handle;
}
