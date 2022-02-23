#include "context_manager.h"

cusolverDnHandle_t get_handle()
{
  static __thread cusolverDnHandle_t handle;

  if(handle == NULL)
  {
    cusolverStatus_t stat;
    stat = cusolverDnCreate(&handle);

    if (stat != CUSOLVER_STATUS_SUCCESS)
    {
      printf("CUSOLVER initialization failed! Status: %d\n", stat);
    }
  }
  return handle;
}
