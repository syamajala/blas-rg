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

  var status = sscal(alpha, a)

  for i in a.ispace do
    c.printf("A[%d] = %0.f\n", i, a[i])
  end

  status = sscal(alpha, a)

  for i in a.ispace do
    c.printf("A[%d] = %0.f\n", i, a[i])
  end
end

regentlib.start(main)
