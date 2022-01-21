copyright = """-- Copyright 2021 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
"""

header = """
import "regent"
local clib = regentlib.c
local nan = regentlib.nan(double)
local utils = require("utils")

float_ptr = raw_ptr_factory(float)
double_ptr = raw_ptr_factory(double)
complex_ptr = raw_ptr_factory(complex)

%s

"""

cblas = """
local cblas = utils.cblas
"""

cblas_header = header % cblas

cublas = """
local cuda_home = os.getenv("CUDA_HOME")
terralib.includepath = terralib.includepath .. ";" .. cuda_home .. "/include"

terralib.linklibrary(cuda_home .. "/lib64/libcublas.so")
terralib.linklibrary("./libcontext_manager.so")

local cuda_runtime = terralib.includec("cuda_runtime.h")
local cublas = terralib.includec("cublas_v2.h")

local mgr = terralib.includec("context_manager.h", {"-I", "../"})
"""

cublas_header = header % cublas

blas = """
local cblas_h = utils.cblas
require("cblas")
if utils.use_gpu then
   require("cublas")
end
"""

blas_header = header % blas

task_template = """
__demand(__cuda, __leaf)
task %s(%s)
where
%s
do
%s
end\n"""

task_template_no_priv = """
__demand(__cuda, __leaf)
task %s(%s)
%s
end\n"""

extern_template = """
extern task %s(%s)
where
%s
end\n"""

extern_template_no_priv = """
extern task %s(%s)\n"""

footer = """local tasks_h = "%s_tasks.h"
local tasks_so = "%s_tasks.so"
regentlib.save_tasks(tasks_h, tasks_so, nil, nil, nil, nil, false)
"""
