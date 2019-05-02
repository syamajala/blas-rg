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


local cuda_home = os.getenv("CUDA_HOME")
terralib.includepath = terralib.includepath .. ";" .. cuda_home .. "/include"

terralib.linklibrary(cuda_home .. "/lib64/libcublas.so")
terralib.linklibrary("./libcontext_manager.so")

local cuda_runtime = terralib.includec("cuda_runtime.h")
local cublas = terralib.includec("cublas_v2.h")

local mgr = terralib.includec("context_manager.h", {"-I", "."})


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

terra snrm2_terra(
	n : int,
	rectX : rect1d,
	result : float,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSnrm2_v2(handle, n, rawX.ptr, rawX.offset, &result)
end

terra dnrm2_terra(
	n : int,
	rectX : rect1d,
	result : double,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDnrm2_v2(handle, n, rawX.ptr, rawX.offset, &result)
end

terra sdot_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	result : float,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSdot_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, &result)
end

terra ddot_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	result : double,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDdot_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, &result)
end

terra sscal_terra(
	n : int,
	alpha : float,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSscal_v2(handle, n, &alpha, rawX.ptr, rawX.offset)
end

terra dscal_terra(
	n : int,
	alpha : double,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDscal_v2(handle, n, &alpha, rawX.ptr, rawX.offset)
end

terra saxpy_terra(
	n : int,
	alpha : float,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSaxpy_v2(handle, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra daxpy_terra(
	n : int,
	alpha : double,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDaxpy_v2(handle, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra scopy_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasScopy_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dcopy_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDcopy_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra sswap_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSswap_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dswap_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDswap_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra isamax_terra(
	n : int,
	rectX : rect1d,
	result : int,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasIsamax_v2(handle, n, rawX.ptr, rawX.offset, &result)
end

terra idamax_terra(
	n : int,
	rectX : rect1d,
	result : int,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasIdamax_v2(handle, n, rawX.ptr, rawX.offset, &result)
end

terra sasum_terra(
	n : int,
	rectX : rect1d,
	result : float,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSasum_v2(handle, n, rawX.ptr, rawX.offset, &result)
end

terra dasum_terra(
	n : int,
	rectX : rect1d,
	result : double,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDasum_v2(handle, n, rawX.ptr, rawX.offset, &result)
end

terra srot_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	c : float,
	s : float,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSrot_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, &c, &s)
end

terra drot_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	c : double,
	s : double,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDrot_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, &c, &s)
end

terra srotg_terra(
	a : float,
	b : float,
	c : float,
	s : float)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSrotg_v2(handle, &a, &b, &c, &s)
end

terra drotg_terra(
	a : double,
	b : double,
	c : double,
	s : double)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDrotg_v2(handle, &a, &b, &c, &s)
end

terra srotm_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	rectPARAM : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t,
	prPARAM : c.legion_physical_region_t,
	fldPARAM : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawPARAM : float_ptr
	[get_raw_ptr_factory(1, float, rectPARAM, prPARAM, fldPARAM, rawPARAM, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSrotm_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawPARAM.ptr)
end

terra drotm_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	rectPARAM : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t,
	prPARAM : c.legion_physical_region_t,
	fldPARAM : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawPARAM : double_ptr
	[get_raw_ptr_factory(1, double, rectPARAM, prPARAM, fldPARAM, rawPARAM, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDrotm_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawPARAM.ptr)
end

terra srotmg_terra(
	d1 : float,
	d2 : float,
	x1 : float,
	y1 : float,
	rectPARAM : rect1d,
	prPARAM : c.legion_physical_region_t,
	fldPARAM : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawPARAM : float_ptr
	[get_raw_ptr_factory(1, float, rectPARAM, prPARAM, fldPARAM, rawPARAM, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSrotmg_v2(handle, &d1, &d2, &x1, &y1, rawPARAM.ptr)
end

terra drotmg_terra(
	d1 : double,
	d2 : double,
	x1 : double,
	y1 : double,
	rectPARAM : rect1d,
	prPARAM : c.legion_physical_region_t,
	fldPARAM : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawPARAM : double_ptr
	[get_raw_ptr_factory(1, double, rectPARAM, prPARAM, fldPARAM, rawPARAM, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDrotmg_v2(handle, &d1, &d2, &x1, &y1, rawPARAM.ptr)
end

terra sgemv_terra(
	trans : int,
	m : int,
	n : int,
	alpha : float,
	rectA : rect2d,
	rectX : rect1d,
	beta : float,
	rectY : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSgemv_v2(handle, trans, m, n, &alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra dgemv_terra(
	trans : int,
	m : int,
	n : int,
	alpha : double,
	rectA : rect2d,
	rectX : rect1d,
	beta : double,
	rectY : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDgemv_v2(handle, trans, m, n, &alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra sgbmv_terra(
	trans : int,
	m : int,
	n : int,
	kl : int,
	ku : int,
	alpha : float,
	rectA : rect2d,
	rectX : rect1d,
	beta : float,
	rectY : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSgbmv_v2(handle, trans, m, n, kl, ku, &alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra dgbmv_terra(
	trans : int,
	m : int,
	n : int,
	kl : int,
	ku : int,
	alpha : double,
	rectA : rect2d,
	rectX : rect1d,
	beta : double,
	rectY : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDgbmv_v2(handle, trans, m, n, kl, ku, &alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra strmv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStrmv_v2(handle, uplo, trans, diag, n, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtrmv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtrmv_v2(handle, uplo, trans, diag, n, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stbmv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	k : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStbmv_v2(handle, uplo, trans, diag, n, k, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtbmv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	k : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtbmv_v2(handle, uplo, trans, diag, n, k, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stpmv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectAP : rect1d,
	rectX : rect1d,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawAP : float_ptr
	[get_raw_ptr_factory(1, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStpmv_v2(handle, uplo, trans, diag, n, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra dtpmv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectAP : rect1d,
	rectX : rect1d,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawAP : double_ptr
	[get_raw_ptr_factory(1, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtpmv_v2(handle, uplo, trans, diag, n, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra strsv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStrsv_v2(handle, uplo, trans, diag, n, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtrsv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtrsv_v2(handle, uplo, trans, diag, n, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stpsv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectAP : rect1d,
	rectX : rect1d,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawAP : float_ptr
	[get_raw_ptr_factory(1, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStpsv_v2(handle, uplo, trans, diag, n, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra dtpsv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectAP : rect1d,
	rectX : rect1d,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawAP : double_ptr
	[get_raw_ptr_factory(1, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtpsv_v2(handle, uplo, trans, diag, n, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra stbsv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	k : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStbsv_v2(handle, uplo, trans, diag, n, k, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtbsv_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	k : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtbsv_v2(handle, uplo, trans, diag, n, k, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra ssymv_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectA : rect2d,
	rectX : rect1d,
	beta : float,
	rectY : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSsymv_v2(handle, uplo, n, &alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra dsymv_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectA : rect2d,
	rectX : rect1d,
	beta : double,
	rectY : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDsymv_v2(handle, uplo, n, &alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra ssbmv_terra(
	uplo : int,
	n : int,
	k : int,
	alpha : float,
	rectA : rect2d,
	rectX : rect1d,
	beta : float,
	rectY : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSsbmv_v2(handle, uplo, n, k, &alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra dsbmv_terra(
	uplo : int,
	n : int,
	k : int,
	alpha : double,
	rectA : rect2d,
	rectX : rect1d,
	beta : double,
	rectY : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDsbmv_v2(handle, uplo, n, k, &alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra sspmv_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectAP : rect1d,
	rectX : rect1d,
	beta : float,
	rectY : rect1d,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawAP : float_ptr
	[get_raw_ptr_factory(1, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSspmv_v2(handle, uplo, n, &alpha, rawAP.ptr, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra dspmv_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectAP : rect1d,
	rectX : rect1d,
	beta : double,
	rectY : rect1d,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawAP : double_ptr
	[get_raw_ptr_factory(1, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDspmv_v2(handle, uplo, n, &alpha, rawAP.ptr, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra sger_terra(
	m : int,
	n : int,
	alpha : float,
	rectX : rect1d,
	rectY : rect1d,
	rectA : rect2d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSger_v2(handle, m, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra dger_terra(
	m : int,
	n : int,
	alpha : double,
	rectX : rect1d,
	rectY : rect1d,
	rectA : rect2d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDger_v2(handle, m, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra ssyr_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectX : rect1d,
	rectA : rect2d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSsyr_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawA.ptr, rawA.offset)
end

terra dsyr_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectX : rect1d,
	rectA : rect2d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDsyr_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawA.ptr, rawA.offset)
end

terra sspr_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectX : rect1d,
	rectAP : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawAP : float_ptr
	[get_raw_ptr_factory(1, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSspr_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawAP.ptr)
end

terra dspr_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectX : rect1d,
	rectAP : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawAP : double_ptr
	[get_raw_ptr_factory(1, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDspr_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawAP.ptr)
end

terra ssyr2_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectX : rect1d,
	rectY : rect1d,
	rectA : rect2d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSsyr2_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra dsyr2_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectX : rect1d,
	rectY : rect1d,
	rectA : rect2d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDsyr2_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra sspr2_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectX : rect1d,
	rectY : rect1d,
	rectAP : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawAP : float_ptr
	[get_raw_ptr_factory(1, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSspr2_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawAP.ptr)
end

terra dspr2_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectX : rect1d,
	rectY : rect1d,
	rectAP : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawAP : double_ptr
	[get_raw_ptr_factory(1, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDspr2_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawAP.ptr)
end

terra sgemm_terra(
	transa : int,
	transb : int,
	m : int,
	n : int,
	k : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	beta : float,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSgemm_v2(handle, transa, transb, m, n, k, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, &beta, rawC.ptr, rawC.offset)
end

terra dgemm_terra(
	transa : int,
	transb : int,
	m : int,
	n : int,
	k : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	beta : double,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDgemm_v2(handle, transa, transb, m, n, k, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, &beta, rawC.ptr, rawC.offset)
end

terra ssyrk_terra(
	uplo : int,
	trans : int,
	n : int,
	k : int,
	alpha : float,
	rectA : rect2d,
	beta : float,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSsyrk_v2(handle, uplo, trans, n, k, &alpha, rawA.ptr, rawA.offset, &beta, rawC.ptr, rawC.offset)
end

terra dsyrk_terra(
	uplo : int,
	trans : int,
	n : int,
	k : int,
	alpha : double,
	rectA : rect2d,
	beta : double,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDsyrk_v2(handle, uplo, trans, n, k, &alpha, rawA.ptr, rawA.offset, &beta, rawC.ptr, rawC.offset)
end

terra ssyr2k_terra(
	uplo : int,
	trans : int,
	n : int,
	k : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	beta : float,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSsyr2k_v2(handle, uplo, trans, n, k, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, &beta, rawC.ptr, rawC.offset)
end

terra dsyr2k_terra(
	uplo : int,
	trans : int,
	n : int,
	k : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	beta : double,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDsyr2k_v2(handle, uplo, trans, n, k, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, &beta, rawC.ptr, rawC.offset)
end

terra ssymm_terra(
	side : int,
	uplo : int,
	m : int,
	n : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	beta : float,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSsymm_v2(handle, side, uplo, m, n, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, &beta, rawC.ptr, rawC.offset)
end

terra dsymm_terra(
	side : int,
	uplo : int,
	m : int,
	n : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	beta : double,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDsymm_v2(handle, side, uplo, m, n, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, &beta, rawC.ptr, rawC.offset)
end

terra strsm_terra(
	side : int,
	uplo : int,
	trans : int,
	diag : int,
	m : int,
	n : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra dtrsm_terra(
	side : int,
	uplo : int,
	trans : int,
	diag : int,
	m : int,
	n : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra strmm_terra(
	side : int,
	uplo : int,
	trans : int,
	diag : int,
	m : int,
	n : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, rawC.ptr, rawC.offset)
end

terra dtrmm_terra(
	side : int,
	uplo : int,
	trans : int,
	diag : int,
	m : int,
	n : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, rawC.ptr, rawC.offset)
end


__demand(__cuda, __leaf)
task snrm2(
	X : region(ispace(int1d), float),
	result : float)
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = (sizeX-0)/1
	return snrm2_terra(n, rectX, result, __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task dnrm2(
	X : region(ispace(int1d), double),
	result : double)
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = (sizeX-0)/1
	return dnrm2_terra(n, rectX, result, __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task sdot(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float),
	result : float)
where
	reads(X),
	reads(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = (sizeX-0)/1
	return sdot_terra(n, rectX, rectY, result, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task ddot(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double),
	result : double)
where
	reads(X),
	reads(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = (sizeX-0)/1
	return ddot_terra(n, rectX, rectY, result, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task sscal(
	alpha : float,
	X : region(ispace(int1d), float))
where
	reads writes(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = (sizeX-0)/1
	return sscal_terra(n, alpha, rectX, __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task dscal(
	alpha : double,
	X : region(ispace(int1d), double))
where
	reads writes(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = (sizeX-0)/1
	return dscal_terra(n, alpha, rectX, __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task saxpy(
	alpha : float,
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = (sizeX-0)/1
	return saxpy_terra(n, alpha, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task daxpy(
	alpha : double,
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = (sizeX-0)/1
	return daxpy_terra(n, alpha, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task scopy(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = (sizeX-0)/1
	return scopy_terra(n, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task dcopy(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = (sizeX-0)/1
	return dcopy_terra(n, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task sswap(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads writes(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = (sizeX-0)/1
	return sswap_terra(n, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task dswap(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads writes(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = (sizeX-0)/1
	return dswap_terra(n, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task isamax(
	X : region(ispace(int1d), float),
	result : int)
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = (sizeX-0)/1
	return isamax_terra(n, rectX, result, __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task idamax(
	X : region(ispace(int1d), double),
	result : int)
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = (sizeX-0)/1
	return idamax_terra(n, rectX, result, __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task sasum(
	X : region(ispace(int1d), float),
	result : float)
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = (sizeX-0)/1
	return sasum_terra(n, rectX, result, __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task dasum(
	X : region(ispace(int1d), double),
	result : double)
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = (sizeX-0)/1
	return dasum_terra(n, rectX, result, __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task srot(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float),
	c : float,
	s : float)
where
	reads writes(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = (sizeX-1-0)/1+1
	return srot_terra(n, rectX, rectY, c, s, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task drot(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double),
	c : double,
	s : double)
where
	reads writes(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = (sizeX-1-0)/1+1
	return drot_terra(n, rectX, rectY, c, s, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task srotg(
	a : float,
	b : float,
	c : float,
	s : float)
	return srotg_terra(a, b, c, s)
end

__demand(__cuda, __leaf)
task drotg(
	a : double,
	b : double,
	c : double,
	s : double)
	return drotg_terra(a, b, c, s)
end

__demand(__cuda, __leaf)
task srotm(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float),
	PARAM : region(ispace(int1d), float))
where
	reads writes(X),
	reads writes(Y),
	reads(PARAM)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var rectPARAM = PARAM.bounds
	var sizePARAM = rectPARAM.hi - rectPARAM.lo + {1}
	var n = (sizeX-0)/1
	return srotm_terra(n, rectX, rectY, rectPARAM, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(PARAM)[0], __fields(PARAM)[0])
end

__demand(__cuda, __leaf)
task drotm(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double),
	PARAM : region(ispace(int1d), double))
where
	reads writes(X),
	reads writes(Y),
	reads(PARAM)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var rectPARAM = PARAM.bounds
	var sizePARAM = rectPARAM.hi - rectPARAM.lo + {1}
	var n = (sizeX-0)/1
	return drotm_terra(n, rectX, rectY, rectPARAM, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(PARAM)[0], __fields(PARAM)[0])
end

__demand(__cuda, __leaf)
task srotmg(
	d1 : float,
	d2 : float,
	x1 : float,
	y1 : float,
	PARAM : region(ispace(int1d), float))
where
	reads writes(PARAM)
do
	var rectPARAM = PARAM.bounds
	var sizePARAM = rectPARAM.hi - rectPARAM.lo + {1}
	return srotmg_terra(d1, d2, x1, y1, rectPARAM, __physical(PARAM)[0], __fields(PARAM)[0])
end

__demand(__cuda, __leaf)
task drotmg(
	d1 : double,
	d2 : double,
	x1 : double,
	y1 : double,
	PARAM : region(ispace(int1d), double))
where
	reads writes(PARAM)
do
	var rectPARAM = PARAM.bounds
	var sizePARAM = rectPARAM.hi - rectPARAM.lo + {1}
	return drotmg_terra(d1, d2, x1, y1, rectPARAM, __physical(PARAM)[0], __fields(PARAM)[0])
end

__demand(__cuda, __leaf)
task sgemv(
	trans : int,
	alpha : float,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float),
	beta : float,
	Y : region(ispace(int1d), float))
where
	reads(A),
	reads(X),
	reads writes(Y)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var m = sizeA.x
	var n = sizeA.y
	return sgemv_terra(trans, m, n, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task dgemv(
	trans : int,
	alpha : double,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double),
	beta : double,
	Y : region(ispace(int1d), double))
where
	reads(A),
	reads(X),
	reads writes(Y)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var m = sizeA.x
	var n = sizeA.y
	return dgemv_terra(trans, m, n, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task sgbmv(
	trans : int,
	m : int,
	n : int,
	kl : int,
	ku : int,
	alpha : float,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float),
	beta : float,
	Y : region(ispace(int1d), float))
where
	reads(A),
	reads(X),
	reads writes(Y)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	return sgbmv_terra(trans, m, n, kl, ku, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task dgbmv(
	trans : int,
	m : int,
	n : int,
	kl : int,
	ku : int,
	alpha : double,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double),
	beta : double,
	Y : region(ispace(int1d), double))
where
	reads(A),
	reads(X),
	reads writes(Y)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	return dgbmv_terra(trans, m, n, kl, ku, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task strmv(
	uplo : int,
	trans : int,
	diag : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = sizeA.x
	return strmv_terra(uplo, trans, diag, n, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task dtrmv(
	uplo : int,
	trans : int,
	diag : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = sizeA.x
	return dtrmv_terra(uplo, trans, diag, n, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task stbmv(
	uplo : int,
	trans : int,
	diag : int,
	k : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = sizeA.y
	return stbmv_terra(uplo, trans, diag, n, k, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task dtbmv(
	uplo : int,
	trans : int,
	diag : int,
	k : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = sizeA.y
	return dtbmv_terra(uplo, trans, diag, n, k, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task stpmv(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	AP : region(ispace(int1d), float),
	X : region(ispace(int1d), float))
where
	reads(AP),
	reads writes(X)
do
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	return stpmv_terra(uplo, trans, diag, n, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task dtpmv(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	AP : region(ispace(int1d), double),
	X : region(ispace(int1d), double))
where
	reads(AP),
	reads writes(X)
do
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	return dtpmv_terra(uplo, trans, diag, n, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task strsv(
	uplo : int,
	trans : int,
	diag : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = sizeA.x
	return strsv_terra(uplo, trans, diag, n, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task dtrsv(
	uplo : int,
	trans : int,
	diag : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = sizeA.x
	return dtrsv_terra(uplo, trans, diag, n, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task stpsv(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	AP : region(ispace(int1d), float),
	X : region(ispace(int1d), float))
where
	reads(AP),
	reads writes(X)
do
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	return stpsv_terra(uplo, trans, diag, n, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task dtpsv(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	AP : region(ispace(int1d), double),
	X : region(ispace(int1d), double))
where
	reads(AP),
	reads writes(X)
do
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	return dtpsv_terra(uplo, trans, diag, n, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task stbsv(
	uplo : int,
	trans : int,
	diag : int,
	k : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = sizeA.y
	return stbsv_terra(uplo, trans, diag, n, k, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task dtbsv(
	uplo : int,
	trans : int,
	diag : int,
	k : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var n = sizeA.y
	return dtbsv_terra(uplo, trans, diag, n, k, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__cuda, __leaf)
task ssymv(
	uplo : int,
	alpha : float,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float),
	beta : float,
	Y : region(ispace(int1d), float))
where
	reads(A),
	reads(X),
	reads writes(Y)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = sizeA.x
	return ssymv_terra(uplo, n, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task dsymv(
	uplo : int,
	alpha : double,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double),
	beta : double,
	Y : region(ispace(int1d), double))
where
	reads(A),
	reads(X),
	reads writes(Y)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = sizeA.x
	return dsymv_terra(uplo, n, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task ssbmv(
	uplo : int,
	k : int,
	alpha : float,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float),
	beta : float,
	Y : region(ispace(int1d), float))
where
	reads(A),
	reads(X),
	reads writes(Y)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = sizeA.y
	return ssbmv_terra(uplo, n, k, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task dsbmv(
	uplo : int,
	k : int,
	alpha : double,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double),
	beta : double,
	Y : region(ispace(int1d), double))
where
	reads(A),
	reads(X),
	reads writes(Y)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var n = sizeA.y
	return dsbmv_terra(uplo, n, k, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task sspmv(
	uplo : int,
	n : int,
	alpha : float,
	AP : region(ispace(int1d), float),
	X : region(ispace(int1d), float),
	beta : float,
	Y : region(ispace(int1d), float))
where
	reads(AP),
	reads(X),
	reads writes(Y)
do
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	return sspmv_terra(uplo, n, alpha, rectAP, rectX, beta, rectY, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task dspmv(
	uplo : int,
	n : int,
	alpha : double,
	AP : region(ispace(int1d), double),
	X : region(ispace(int1d), double),
	beta : double,
	Y : region(ispace(int1d), double))
where
	reads(AP),
	reads(X),
	reads writes(Y)
do
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	return dspmv_terra(uplo, n, alpha, rectAP, rectX, beta, rectY, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__cuda, __leaf)
task sger(
	alpha : float,
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float),
	A : region(ispace(int2d), float))
where
	reads(X),
	reads(Y),
	reads writes(A)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var m = sizeX
	var n = sizeY
	return sger_terra(m, n, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__cuda, __leaf)
task dger(
	alpha : double,
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double),
	A : region(ispace(int2d), double))
where
	reads(X),
	reads(Y),
	reads writes(A)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var m = sizeX
	var n = sizeY
	return dger_terra(m, n, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__cuda, __leaf)
task ssyr(
	uplo : int,
	alpha : float,
	X : region(ispace(int1d), float),
	A : region(ispace(int2d), float))
where
	reads(X),
	reads writes(A)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var n = (sizeX-1-0)/1+1
	return ssyr_terra(uplo, n, alpha, rectX, rectA, __physical(X)[0], __fields(X)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__cuda, __leaf)
task dsyr(
	uplo : int,
	alpha : double,
	X : region(ispace(int1d), double),
	A : region(ispace(int2d), double))
where
	reads(X),
	reads writes(A)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var n = (sizeX-1-0)/1+1
	return dsyr_terra(uplo, n, alpha, rectX, rectA, __physical(X)[0], __fields(X)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__cuda, __leaf)
task sspr(
	uplo : int,
	n : int,
	alpha : float,
	X : region(ispace(int1d), float),
	AP : region(ispace(int1d), float))
where
	reads(X),
	reads writes(AP)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1}
	return sspr_terra(uplo, n, alpha, rectX, rectAP, __physical(X)[0], __fields(X)[0], __physical(AP)[0], __fields(AP)[0])
end

__demand(__cuda, __leaf)
task dspr(
	uplo : int,
	n : int,
	alpha : double,
	X : region(ispace(int1d), double),
	AP : region(ispace(int1d), double))
where
	reads(X),
	reads writes(AP)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1}
	return dspr_terra(uplo, n, alpha, rectX, rectAP, __physical(X)[0], __fields(X)[0], __physical(AP)[0], __fields(AP)[0])
end

__demand(__cuda, __leaf)
task ssyr2(
	uplo : int,
	alpha : float,
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float),
	A : region(ispace(int2d), float))
where
	reads(X),
	reads(Y),
	reads writes(A)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var n = 0
	if [bool]((sizeX-1-0)/1+1 <=(sizeY-1-0)/1+1) then
		n = (sizeX-1-0)/1+1
	else
		n = (sizeY-1-0)/1+1
	end

	return ssyr2_terra(uplo, n, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__cuda, __leaf)
task dsyr2(
	uplo : int,
	alpha : double,
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double),
	A : region(ispace(int2d), double))
where
	reads(X),
	reads(Y),
	reads writes(A)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var n = 0
	if [bool]((sizeX-1-0)/1+1 <=(sizeY-1-0)/1+1) then
		n = (sizeX-1-0)/1+1
	else
		n = (sizeY-1-0)/1+1
	end

	return dsyr2_terra(uplo, n, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__cuda, __leaf)
task sspr2(
	uplo : int,
	n : int,
	alpha : float,
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float),
	AP : region(ispace(int1d), float))
where
	reads(X),
	reads(Y),
	reads writes(AP)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1}
	return sspr2_terra(uplo, n, alpha, rectX, rectY, rectAP, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(AP)[0], __fields(AP)[0])
end

__demand(__cuda, __leaf)
task dspr2(
	uplo : int,
	n : int,
	alpha : double,
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double),
	AP : region(ispace(int1d), double))
where
	reads(X),
	reads(Y),
	reads writes(AP)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1}
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1}
	return dspr2_terra(uplo, n, alpha, rectX, rectY, rectAP, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(AP)[0], __fields(AP)[0])
end

__demand(__cuda, __leaf)
task sgemm(
	transa : int,
	transb : int,
	alpha : float,
	A : region(ispace(int2d), float),
	B : region(ispace(int2d), float),
	beta : float,
	C : region(ispace(int2d), float))
where
	reads(A),
	reads(B),
	reads writes(C)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectB = B.bounds
	var sizeB = rectB.hi - rectB.lo + {1, 1}
	var rectC = C.bounds
	var sizeC = rectC.hi - rectC.lo + {1, 1}
	var m = 0
	if [bool](transa) then
		m = sizeA.y
	else
		m = sizeA.x
	end

	var n = 0
	if [bool](transb) then
		n = sizeB.x
	else
		n = sizeB.y
	end

	var k = 0
	if [bool](transa) then
		k = sizeA.x
	else
		k = sizeA.y
	end

	return sgemm_terra(transa, transb, m, n, k, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__cuda, __leaf)
task dgemm(
	transa : int,
	transb : int,
	alpha : double,
	A : region(ispace(int2d), double),
	B : region(ispace(int2d), double),
	beta : double,
	C : region(ispace(int2d), double))
where
	reads(A),
	reads(B),
	reads writes(C)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectB = B.bounds
	var sizeB = rectB.hi - rectB.lo + {1, 1}
	var rectC = C.bounds
	var sizeC = rectC.hi - rectC.lo + {1, 1}
	var m = 0
	if [bool](transa) then
		m = sizeA.y
	else
		m = sizeA.x
	end

	var n = 0
	if [bool](transb) then
		n = sizeB.x
	else
		n = sizeB.y
	end

	var k = 0
	if [bool](transa) then
		k = sizeA.x
	else
		k = sizeA.y
	end

	return dgemm_terra(transa, transb, m, n, k, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__cuda, __leaf)
task ssyrk(
	uplo : int,
	trans : int,
	alpha : float,
	A : region(ispace(int2d), float),
	beta : float,
	C : region(ispace(int2d), float))
where
	reads(A),
	reads writes(C)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectC = C.bounds
	var sizeC = rectC.hi - rectC.lo + {1, 1}
	var n = 0
	if [bool](trans) then
		n = sizeA.y
	else
		n = sizeA.x
	end

	var k = 0
	if [bool](trans) then
		k = sizeA.x
	else
		k = sizeA.y
	end

	return ssyrk_terra(uplo, trans, n, k, alpha, rectA, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__cuda, __leaf)
task dsyrk(
	uplo : int,
	trans : int,
	alpha : double,
	A : region(ispace(int2d), double),
	beta : double,
	C : region(ispace(int2d), double))
where
	reads(A),
	reads writes(C)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectC = C.bounds
	var sizeC = rectC.hi - rectC.lo + {1, 1}
	var n = 0
	if [bool](trans) then
		n = sizeA.y
	else
		n = sizeA.x
	end

	var k = 0
	if [bool](trans) then
		k = sizeA.x
	else
		k = sizeA.y
	end

	return dsyrk_terra(uplo, trans, n, k, alpha, rectA, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__cuda, __leaf)
task ssyr2k(
	uplo : int,
	trans : int,
	alpha : float,
	A : region(ispace(int2d), float),
	B : region(ispace(int2d), float),
	beta : float,
	C : region(ispace(int2d), float))
where
	reads(A),
	reads(B),
	reads writes(C)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectB = B.bounds
	var sizeB = rectB.hi - rectB.lo + {1, 1}
	var rectC = C.bounds
	var sizeC = rectC.hi - rectC.lo + {1, 1}
	var n = 0
	if [bool](trans) then
		n = sizeA.y
	else
		n = sizeA.x
	end

	var k = 0
	if [bool](trans) then
		k = sizeA.x
	else
		k = sizeA.y
	end

	return ssyr2k_terra(uplo, trans, n, k, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__cuda, __leaf)
task dsyr2k(
	uplo : int,
	trans : int,
	alpha : double,
	A : region(ispace(int2d), double),
	B : region(ispace(int2d), double),
	beta : double,
	C : region(ispace(int2d), double))
where
	reads(A),
	reads(B),
	reads writes(C)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectB = B.bounds
	var sizeB = rectB.hi - rectB.lo + {1, 1}
	var rectC = C.bounds
	var sizeC = rectC.hi - rectC.lo + {1, 1}
	var n = 0
	if [bool](trans) then
		n = sizeA.y
	else
		n = sizeA.x
	end

	var k = 0
	if [bool](trans) then
		k = sizeA.x
	else
		k = sizeA.y
	end

	return dsyr2k_terra(uplo, trans, n, k, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__cuda, __leaf)
task ssymm(
	side : int,
	uplo : int,
	alpha : float,
	A : region(ispace(int2d), float),
	B : region(ispace(int2d), float),
	beta : float,
	C : region(ispace(int2d), float))
where
	reads(A),
	reads(B),
	reads writes(C)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectB = B.bounds
	var sizeB = rectB.hi - rectB.lo + {1, 1}
	var rectC = C.bounds
	var sizeC = rectC.hi - rectC.lo + {1, 1}
	var m = 0
	if [bool](side) then
		m = sizeB.x
	else
		m = sizeA.x
	end

	var n = 0
	if [bool](side) then
		n = sizeA.y
	else
		n = sizeB.y
	end

	return ssymm_terra(side, uplo, m, n, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__cuda, __leaf)
task dsymm(
	side : int,
	uplo : int,
	alpha : double,
	A : region(ispace(int2d), double),
	B : region(ispace(int2d), double),
	beta : double,
	C : region(ispace(int2d), double))
where
	reads(A),
	reads(B),
	reads writes(C)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectB = B.bounds
	var sizeB = rectB.hi - rectB.lo + {1, 1}
	var rectC = C.bounds
	var sizeC = rectC.hi - rectC.lo + {1, 1}
	var m = 0
	if [bool](side) then
		m = sizeB.x
	else
		m = sizeA.x
	end

	var n = 0
	if [bool](side) then
		n = sizeA.y
	else
		n = sizeB.y
	end

	return dsymm_terra(side, uplo, m, n, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__cuda, __leaf)
task strsm(
	side : int,
	uplo : int,
	trans : int,
	diag : int,
	alpha : float,
	A : region(ispace(int2d), float),
	B : region(ispace(int2d), float))
where
	reads(A),
	reads writes(B)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectB = B.bounds
	var sizeB = rectB.hi - rectB.lo + {1, 1}
	var m = sizeB.x
	var n = sizeB.y
	return strsm_terra(side, uplo, trans, diag, m, n, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
end

__demand(__cuda, __leaf)
task dtrsm(
	side : int,
	uplo : int,
	trans : int,
	diag : int,
	alpha : double,
	A : region(ispace(int2d), double),
	B : region(ispace(int2d), double))
where
	reads(A),
	reads writes(B)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectB = B.bounds
	var sizeB = rectB.hi - rectB.lo + {1, 1}
	var m = sizeB.x
	var n = sizeB.y
	return dtrsm_terra(side, uplo, trans, diag, m, n, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
end

__demand(__cuda, __leaf)
task strmm(
	side : int,
	uplo : int,
	trans : int,
	diag : int,
	alpha : float,
	A : region(ispace(int2d), float),
	B : region(ispace(int2d), float),
	C : region(ispace(int2d), float))
where
	reads(A),
	reads writes(B),
	reads writes(C)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectB = B.bounds
	var sizeB = rectB.hi - rectB.lo + {1, 1}
	var rectC = C.bounds
	var sizeC = rectC.hi - rectC.lo + {1, 1}
	var m = 0
	if [bool](side) then
		m = sizeB.y
	else
		m = sizeA.y
	end

	var n = sizeB.y
	return strmm_terra(side, uplo, trans, diag, m, n, alpha, rectA, rectB, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__cuda, __leaf)
task dtrmm(
	side : int,
	uplo : int,
	trans : int,
	diag : int,
	alpha : double,
	A : region(ispace(int2d), double),
	B : region(ispace(int2d), double),
	C : region(ispace(int2d), double))
where
	reads(A),
	reads writes(B),
	reads writes(C)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectB = B.bounds
	var sizeB = rectB.hi - rectB.lo + {1, 1}
	var rectC = C.bounds
	var sizeC = rectC.hi - rectC.lo + {1, 1}
	var m = 0
	if [bool](side) then
		m = sizeB.y
	else
		m = sizeA.y
	end

	var n = sizeB.y
	return dtrmm_terra(side, uplo, trans, diag, m, n, alpha, rectA, rectB, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end
