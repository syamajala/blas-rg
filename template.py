copyright = """-- Copyright 2019 Stanford University
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
local c = regentlib.c

%s

function raw_ptr_factory(typ)
  local struct raw_ptr
  {
    ptr : &typ,
    offset : int,
  }
  return raw_ptr
end

local float_ptr = raw_ptr_factory(float)
local double_ptr = raw_ptr_factory(double)
local complex_ptr = raw_ptr_factory(complex)

function get_raw_ptr_factory(dim, typ, rect, pr, fld, raw, raw_ptr)
  if dim == 2 then
    return quote
      var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
      var subrect : c.legion_rect_2d_t
      var offsets : c.legion_byte_offset_t[dim]
      var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
      raw = raw_ptr { ptr = [&typ](ptr), offset = offsets[dim-1].offset / sizeof(typ) }
    end
  elseif dim == 1 then
    return quote
      var fa = c.legion_physical_region_get_field_accessor_array_1d(pr, fld)
      var subrect : c.legion_rect_1d_t
      var offsets : c.legion_byte_offset_t[dim]
      var ptr = c.legion_accessor_array_1d_raw_rect_ptr(fa, rect, &subrect, offsets)
      raw = raw_ptr { ptr = [&typ](ptr), offset = offsets[dim-1].offset / sizeof(typ) }
    end
  end
end

"""

blas = """
terralib.linklibrary("libcblas.so")
local cblas = terralib.includec("cblas.h")
"""

blas_header = header % blas

cublas = """
terralib.includepath = terralib.includepath .. ";/opt/cuda/include/"
terralib.linklibrary("/opt/cuda/lib64/libcublas.so")
terralib.linklibrary("./libcontext_manager.so")

local cuda_runtime = terralib.includec("cuda_runtime.h")
local cublas = terralib.includec("cublas_v2.h")

local mgr = terralib.includec("context_manager.h", {"-I", "."})
"""

cublas_header = header % cublas

task_template = """
__demand(__leaf)
task %s(%s)
where
%s
do
%s
end\n"""

task_template_no_priv = """
__demand(__leaf)
task %s(%s)
%s
end\n"""

cuda_task_template = """
__demand(__cuda, __leaf)
task %s(%s)
where
%s
do
%s
end\n"""

cuda_task_template_no_priv = """
__demand(__cuda, __leaf)
task %s(%s)
%s
end\n"""
