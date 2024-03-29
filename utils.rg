import "regent"
local c = regentlib.c

local utils = {}

utils.use_gpu = nil
utils.blas_header = nil
utils.blas_library = nil
utils.output_dir = "./"

for k, v in pairs(arg) do
  if v == "--use-gpu" then
    utils.use_gpu = true
  elseif v == "--blas-header" then
    utils.blas_header = arg[k+1]
  elseif v == "--blas-library" then
    utils.blas_library = arg[k+1]
  elseif v == "--output-dir" then
    utils.output_dir = arg[k+1]
  end
end

assert(utils.blas_header ~= nil, "Missing BLAS header.")
assert(utils.blas_library ~= nil, "Missing BLAS library.")
terralib.linklibrary(utils.blas_library)
utils.cblas = terralib.includec("cblas.h", {"-I", utils.blas_header})

function raw_ptr_factory(typ)
  local struct raw_ptr
  {
    ptr : &typ,
    offset : int,
  }
  return raw_ptr
end

utils.float_ptr = raw_ptr_factory(float)
utils.double_ptr = raw_ptr_factory(double)
utils.complex_ptr = raw_ptr_factory(complex)

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


terra get_executing_processor(runtime : c.legion_runtime_t)
  var ctx = c.legion_runtime_get_context()
  var result = c.legion_runtime_get_executing_processor(runtime, ctx)
  c.legion_context_destroy(ctx)
  return result
end


return utils
