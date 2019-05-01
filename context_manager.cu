#include "context_manager.h"

void create_handle(cublasHandle_t *handle)
{
  cublasStatus_t stat;
  stat = cublasCreate(handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf("CUBLAS initialization failed!\n");
  }
}
