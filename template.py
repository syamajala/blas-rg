header = """
-- Copyright 2018 Stanford University
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

import "regent"
local c = regentlib.c

terralib.linklibrary("libcblas.so")
local cblas = terralib.includec("cblas.h")

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
end

local raw_ptr = raw_ptr_factory(double)

terra get_raw_ptr_2d(rect: rect2d,
                     pr : c.legion_physical_region_t,
                     fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra get_raw_ptr_1d(rect: rect1d,
                     pr : c.legion_physical_region_t,
                     fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_1d(pr, fld)
  var subrect : c.legion_rect_1d_t
  var offsets : c.legion_byte_offset_t[1]
  var ptr = c.legion_accessor_array_1d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[0].offset / sizeof(double) }
end

"""
