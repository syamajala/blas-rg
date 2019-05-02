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
require("cublas")

task main()

  var a = region(ispace(int1d, int1d{6}), float)
  fill(a, 1)
  var alpha = 2

  for i in a.ispace do
    c.printf("A[%d] = %0.f\n", i, a[i])
  end
  c.printf("\n")

  var status = sscal_gpu(alpha, a)

  for i in a.ispace do
    c.printf("A[%d] = %0.f\n", i, a[i])
  end
  c.printf("\n")

  status = sscal_gpu(alpha, a)

  for i in a.ispace do
    c.printf("A[%d] = %0.f\n", i, a[i])
  end
end

regentlib.start(main)
