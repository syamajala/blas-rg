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


local cuda_home = os.getenv("CUDA_HOME")
terralib.includepath = terralib.includepath .. ";" .. cuda_home .. "/include"

terralib.linklibrary(cuda_home .. "/lib64/libcublas.so")
terralib.linklibrary(utils.output_dir .. "/libblas_context_manager.so")

local cuda_runtime = terralib.includec("cuda_runtime.h")
local cublas = terralib.includec("cublas_v2.h")

local mgr = terralib.includec("blas_context_manager.h", {"-I", "./"})


terra snrm2_gpu_terra(
	n : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var result : float
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	cublas.cublasSnrm2_v2(handle, n, rawX.ptr, rawX.offset, &result)
	return result
end

terra dnrm2_gpu_terra(
	n : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var result : double
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	cublas.cublasDnrm2_v2(handle, n, rawX.ptr, rawX.offset, &result)
	return result
end

terra sdot_gpu_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var result : float
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	cublas.cublasSdot_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, &result)
	return result
end

terra ddot_gpu_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var result : double
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	cublas.cublasDdot_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, &result)
	return result
end

terra sscal_gpu_terra(
	n : int,
	alpha : float,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSscal_v2(handle, n, &alpha, rawX.ptr, rawX.offset)
end

terra dscal_gpu_terra(
	n : int,
	alpha : double,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDscal_v2(handle, n, &alpha, rawX.ptr, rawX.offset)
end

terra saxpy_gpu_terra(
	n : int,
	alpha : float,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSaxpy_v2(handle, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra daxpy_gpu_terra(
	n : int,
	alpha : double,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDaxpy_v2(handle, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra scopy_gpu_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasScopy_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dcopy_gpu_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDcopy_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra sswap_gpu_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSswap_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dswap_gpu_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDswap_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra isamax_gpu_terra(
	n : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var result : int
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	cublas.cublasIsamax_v2(handle, n, rawX.ptr, rawX.offset, &result)
	return result
end

terra idamax_gpu_terra(
	n : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var result : int
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	cublas.cublasIdamax_v2(handle, n, rawX.ptr, rawX.offset, &result)
	return result
end

terra sasum_gpu_terra(
	n : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var result : float
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	cublas.cublasSasum_v2(handle, n, rawX.ptr, rawX.offset, &result)
	return result
end

terra dasum_gpu_terra(
	n : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var result : double
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	cublas.cublasDasum_v2(handle, n, rawX.ptr, rawX.offset, &result)
	return result
end

terra srot_gpu_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	c : float,
	s : float,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSrot_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, &c, &s)
end

terra drot_gpu_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	c : double,
	s : double,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDrot_v2(handle, n, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, &c, &s)
end

terra srotg_gpu_terra(
	a : float,
	b : float,
	c : float,
	s : float)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSrotg_v2(handle, &a, &b, &c, &s)
end

terra drotg_gpu_terra(
	a : double,
	b : double,
	c : double,
	s : double)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDrotg_v2(handle, &a, &b, &c, &s)
end

terra srotm_gpu_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	rectPARAM : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t,
	prPARAM : clib.legion_physical_region_t,
	fldPARAM : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra drotm_gpu_terra(
	n : int,
	rectX : rect1d,
	rectY : rect1d,
	rectPARAM : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t,
	prPARAM : clib.legion_physical_region_t,
	fldPARAM : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra srotmg_gpu_terra(
	d1 : float,
	d2 : float,
	x1 : float,
	y1 : float,
	rectPARAM : rect1d,
	prPARAM : clib.legion_physical_region_t,
	fldPARAM : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawPARAM : float_ptr
	[get_raw_ptr_factory(1, float, rectPARAM, prPARAM, fldPARAM, rawPARAM, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSrotmg_v2(handle, &d1, &d2, &x1, &y1, rawPARAM.ptr)
end

terra drotmg_gpu_terra(
	d1 : double,
	d2 : double,
	x1 : double,
	y1 : double,
	rectPARAM : rect1d,
	prPARAM : clib.legion_physical_region_t,
	fldPARAM : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawPARAM : double_ptr
	[get_raw_ptr_factory(1, double, rectPARAM, prPARAM, fldPARAM, rawPARAM, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDrotmg_v2(handle, &d1, &d2, &x1, &y1, rawPARAM.ptr)
end

terra sgemv_gpu_terra(
	trans : int,
	m : int,
	n : int,
	alpha : float,
	rectA : rect2d,
	rectX : rect1d,
	beta : float,
	rectY : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra dgemv_gpu_terra(
	trans : int,
	m : int,
	n : int,
	alpha : double,
	rectA : rect2d,
	rectX : rect1d,
	beta : double,
	rectY : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra sgbmv_gpu_terra(
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
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra dgbmv_gpu_terra(
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
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra strmv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStrmv_v2(handle, uplo, trans, diag, n, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtrmv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtrmv_v2(handle, uplo, trans, diag, n, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stbmv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	k : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStbmv_v2(handle, uplo, trans, diag, n, k, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtbmv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	k : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtbmv_v2(handle, uplo, trans, diag, n, k, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stpmv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectAP : rect2d,
	rectX : rect1d,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawAP : float_ptr
	[get_raw_ptr_factory(2, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStpmv_v2(handle, uplo, trans, diag, n, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra dtpmv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectAP : rect2d,
	rectX : rect1d,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawAP : double_ptr
	[get_raw_ptr_factory(2, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtpmv_v2(handle, uplo, trans, diag, n, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra strsv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStrsv_v2(handle, uplo, trans, diag, n, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtrsv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtrsv_v2(handle, uplo, trans, diag, n, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stpsv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectAP : rect2d,
	rectX : rect1d,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawAP : float_ptr
	[get_raw_ptr_factory(2, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStpsv_v2(handle, uplo, trans, diag, n, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra dtpsv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	rectAP : rect2d,
	rectX : rect1d,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawAP : double_ptr
	[get_raw_ptr_factory(2, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtpsv_v2(handle, uplo, trans, diag, n, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra stbsv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	k : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStbsv_v2(handle, uplo, trans, diag, n, k, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtbsv_gpu_terra(
	uplo : int,
	trans : int,
	diag : int,
	n : int,
	k : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtbsv_v2(handle, uplo, trans, diag, n, k, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra ssymv_gpu_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectA : rect2d,
	rectX : rect1d,
	beta : float,
	rectY : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra dsymv_gpu_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectA : rect2d,
	rectX : rect1d,
	beta : double,
	rectY : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra ssbmv_gpu_terra(
	uplo : int,
	n : int,
	k : int,
	alpha : float,
	rectA : rect2d,
	rectX : rect1d,
	beta : float,
	rectY : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra dsbmv_gpu_terra(
	uplo : int,
	n : int,
	k : int,
	alpha : double,
	rectA : rect2d,
	rectX : rect1d,
	beta : double,
	rectY : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra sspmv_gpu_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectAP : rect2d,
	rectX : rect1d,
	beta : float,
	rectY : rect1d,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawAP : float_ptr
	[get_raw_ptr_factory(2, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSspmv_v2(handle, uplo, n, &alpha, rawAP.ptr, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra dspmv_gpu_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectAP : rect2d,
	rectX : rect1d,
	beta : double,
	rectY : rect1d,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawAP : double_ptr
	[get_raw_ptr_factory(2, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDspmv_v2(handle, uplo, n, &alpha, rawAP.ptr, rawX.ptr, rawX.offset, &beta, rawY.ptr, rawY.offset)
end

terra sger_gpu_terra(
	m : int,
	n : int,
	alpha : float,
	rectX : rect1d,
	rectY : rect1d,
	rectA : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra dger_gpu_terra(
	m : int,
	n : int,
	alpha : double,
	rectX : rect1d,
	rectY : rect1d,
	rectA : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra ssyr_gpu_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectX : rect1d,
	rectA : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSsyr_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawA.ptr, rawA.offset)
end

terra dsyr_gpu_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectX : rect1d,
	rectA : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDsyr_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawA.ptr, rawA.offset)
end

terra sspr_gpu_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectX : rect1d,
	rectAP : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawAP : float_ptr
	[get_raw_ptr_factory(2, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSspr_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawAP.ptr)
end

terra dspr_gpu_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectX : rect1d,
	rectAP : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawAP : double_ptr
	[get_raw_ptr_factory(2, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDspr_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawAP.ptr)
end

terra ssyr2_gpu_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectX : rect1d,
	rectY : rect1d,
	rectA : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra dsyr2_gpu_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectX : rect1d,
	rectY : rect1d,
	rectA : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra sspr2_gpu_terra(
	uplo : int,
	n : int,
	alpha : float,
	rectX : rect1d,
	rectY : rect1d,
	rectAP : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawAP : float_ptr
	[get_raw_ptr_factory(2, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSspr2_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawAP.ptr)
end

terra dspr2_gpu_terra(
	uplo : int,
	n : int,
	alpha : double,
	rectX : rect1d,
	rectY : rect1d,
	rectAP : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawAP : double_ptr
	[get_raw_ptr_factory(2, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDspr2_v2(handle, uplo, n, &alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawAP.ptr)
end

terra sgemm_gpu_terra(
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
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra dgemm_gpu_terra(
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
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra ssyrk_gpu_terra(
	uplo : int,
	trans : int,
	n : int,
	k : int,
	alpha : float,
	rectA : rect2d,
	beta : float,
	rectC : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasSsyrk_v2(handle, uplo, trans, n, k, &alpha, rawA.ptr, rawA.offset, &beta, rawC.ptr, rawC.offset)
end

terra dsyrk_gpu_terra(
	uplo : int,
	trans : int,
	n : int,
	k : int,
	alpha : double,
	rectA : rect2d,
	beta : double,
	rectC : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDsyrk_v2(handle, uplo, trans, n, k, &alpha, rawA.ptr, rawA.offset, &beta, rawC.ptr, rawC.offset)
end

terra ssyr2k_gpu_terra(
	uplo : int,
	trans : int,
	n : int,
	k : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	beta : float,
	rectC : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra dsyr2k_gpu_terra(
	uplo : int,
	trans : int,
	n : int,
	k : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	beta : double,
	rectC : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra ssymm_gpu_terra(
	side : int,
	uplo : int,
	m : int,
	n : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	beta : float,
	rectC : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra dsymm_gpu_terra(
	side : int,
	uplo : int,
	m : int,
	n : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	beta : double,
	rectC : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra strsm_gpu_terra(
	side : int,
	uplo : int,
	trans : int,
	diag : int,
	m : int,
	n : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra dtrsm_gpu_terra(
	side : int,
	uplo : int,
	trans : int,
	diag : int,
	m : int,
	n : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var stream : cuda_runtime.cudaStream_t
        cuda_runtime.cudaStreamCreate(&stream)
        cublas.cublasSetStream_v2(handle, stream)
	return cublas.cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, &alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra strmm_gpu_terra(
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
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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

terra dtrmm_gpu_terra(
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
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var handle : cublas.cublasHandle_t = mgr.get_blas_handle()
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
