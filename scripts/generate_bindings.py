from generate_bindings_cblas import generate_cpu_bindings
from generate_bindings_cublas import generate_gpu_bindings

gpu_bindings = generate_gpu_bindings()
cpu_bindings = generate_cpu_bindings(gpu_bindings)
