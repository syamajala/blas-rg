-- Copyright 2022 Stanford University
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
local clib = regentlib.c
local nan = regentlib.nan(double)
local utils = require("utils")

float_ptr = raw_ptr_factory(float)
double_ptr = raw_ptr_factory(double)
complex_ptr = raw_ptr_factory(complex)


local lapacke = utils.lapacke

terra dgesvd_cpu_terra(
    layout : int,
    jobu   : &int8,
    jobvt  : &int8,
    M      : int,
    N      : int,
	rectA  : rect2d,
    prA    : clib.legion_physical_region_t,
	fldA   : clib.legion_field_id_t,
    rectS  : rect1d,
    prS    : clib.legion_physical_region_t,
	fldS   : clib.legion_field_id_t,
    rectU  : rect2d,
    prU    : clib.legion_physical_region_t,
	fldU   : clib.legion_field_id_t,
    rectVT : rect2d,
    prVT   : clib.legion_physical_region_t,
	fldVT  : clib.legion_field_id_t)

  var rawA : double_ptr
  [get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]

  var rawS : double_ptr
  [get_raw_ptr_factory(1, double, rectS, prS, fldS, rawS, double_ptr)]

  var rawU : double_ptr
  [get_raw_ptr_factory(2, double, rectU, prU, fldU, rawU, double_ptr)]

  var rawVT : double_ptr
  [get_raw_ptr_factory(2, double, rectVT, prVT, fldVT, rawVT, double_ptr)]

  var size = M
  if M > N then
    size = N
  end

  var superb = [&double](clib.malloc(sizeof(double) * size))

  var ret = lapacke.LAPACKE_dgesvd(layout, @jobu, @jobvt, M, N, rawA.ptr, rawA.offset, rawS.ptr, rawU.ptr, rawU.offset, rawVT.ptr, rawVT.offset, superb)

  clib.free(superb)
  return ret
end


local tasks_h = "lapacke_tasks.h"
local tasks_so = "lapacke_tasks.so"
regentlib.save_tasks(tasks_h, tasks_so, nil, nil, nil, nil, false)
