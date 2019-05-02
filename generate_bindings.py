from template import copyright
from generate_bindings_cblas import generate_cpu_bindings
from generate_bindings_cublas import generate_gpu_bindings

# cpu_bindings = generate_cpu_bindings()
# gpu_bindings = generate_gpu_bindings()

variants = []
for task in cpu_bindings:
    if task not in gpu_bindings:
        continue
    variants.append(task)

with open('blas.rg', 'w') as f:
    f.write(copyright)
    f.write('require("cblas")\n')
    f.write('require("cublas")\n')

    for variant in variants:
        gpu_variant = variant + "_gpu"
        f.write(f"{variant}:set_cuda_variant({gpu_variant}:get_cuda_variant())\n")
