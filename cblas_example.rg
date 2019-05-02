-- Copyright 2019 Stanford University
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
local cblas = terralib.includec("cblas.h")
require("cblas")

task main()

  --- Make sure to use single threaded blas
  cblas.openblas_set_num_threads(1)

  var X = region(ispace(int1d, 4), float)
  var Y = region(ispace(int1d, 4), float)

  X[0] = 1
  X[1] = 1
  X[2] = 0
  X[3] = 0

  Y[0] = 1
  Y[1] = 3
  Y[2] = 0
  Y[3] = 0

  var d = sdot(X, Y)
  c.printf("%f\n", d)

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

  C[{1, 1}] = 0.5

  dgemm(cblas.CblasNoTrans, cblas.CblasTrans, 1, A, B, 2, C)

  for i in C.ispace do
    c.printf("C[%d, %d] = %0.3f\n", i.x, i.y, C[i])
  end

end

regentlib.start(main)
