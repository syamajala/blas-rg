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


local cblas = utils.cblas


terra sdot_cpu_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	return cblas.cblas_sdot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra ddot_cpu_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	return cblas.cblas_ddot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra snrm2_cpu_terra(
	N : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	return cblas.cblas_snrm2(N, rawX.ptr, rawX.offset)
end

terra sasum_cpu_terra(
	N : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	return cblas.cblas_sasum(N, rawX.ptr, rawX.offset)
end

terra dnrm2_cpu_terra(
	N : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	return cblas.cblas_dnrm2(N, rawX.ptr, rawX.offset)
end

terra dasum_cpu_terra(
	N : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	return cblas.cblas_dasum(N, rawX.ptr, rawX.offset)
end

terra isamax_cpu_terra(
	N : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	return cblas.cblas_isamax(N, rawX.ptr, rawX.offset)
end

terra idamax_cpu_terra(
	N : int,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	return cblas.cblas_idamax(N, rawX.ptr, rawX.offset)
end

terra sswap_cpu_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_sswap(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra scopy_cpu_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_scopy(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra saxpy_cpu_terra(
	N : int,
	alpha : float,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_saxpy(N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dswap_cpu_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dswap(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dcopy_cpu_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dcopy(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra daxpy_cpu_terra(
	N : int,
	alpha : double,
	rectX : rect1d,
	rectY : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_daxpy(N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra srotg_cpu_terra(
	a : float,
	b : float,
	c : float,
	s : float)

	cblas.cblas_srotg(&a, &b, &c, &s)
end

terra srotmg_cpu_terra(
	d1 : float,
	d2 : float,
	b1 : float,
	b2 : float,
	rectP : rect1d,
	prP : clib.legion_physical_region_t,
	fldP : clib.legion_field_id_t)

	var rawP : float_ptr
	[get_raw_ptr_factory(1, float, rectP, prP, fldP, rawP, float_ptr)]
	cblas.cblas_srotmg(&d1, &d2, &b1, b2, rawP.ptr)
end

terra srot_cpu_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	c : float,
	s : float,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_srot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, c, s)
end

terra srotm_cpu_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	rectP : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t,
	prP : clib.legion_physical_region_t,
	fldP : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawP : float_ptr
	[get_raw_ptr_factory(1, float, rectP, prP, fldP, rawP, float_ptr)]
	cblas.cblas_srotm(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawP.ptr)
end

terra drotg_cpu_terra(
	a : double,
	b : double,
	c : double,
	s : double)

	cblas.cblas_drotg(&a, &b, &c, &s)
end

terra drotmg_cpu_terra(
	d1 : double,
	d2 : double,
	b1 : double,
	b2 : double,
	rectP : rect1d,
	prP : clib.legion_physical_region_t,
	fldP : clib.legion_field_id_t)

	var rawP : double_ptr
	[get_raw_ptr_factory(1, double, rectP, prP, fldP, rawP, double_ptr)]
	cblas.cblas_drotmg(&d1, &d2, &b1, b2, rawP.ptr)
end

terra drot_cpu_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	c : double,
	s : double,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_drot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, c, s)
end

terra drotm_cpu_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	rectP : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prY : clib.legion_physical_region_t,
	fldY : clib.legion_field_id_t,
	prP : clib.legion_physical_region_t,
	fldP : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawP : double_ptr
	[get_raw_ptr_factory(1, double, rectP, prP, fldP, rawP, double_ptr)]
	cblas.cblas_drotm(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawP.ptr)
end

terra sscal_cpu_terra(
	N : int,
	alpha : float,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_sscal(N, alpha, rawX.ptr, rawX.offset)
end

terra dscal_cpu_terra(
	N : int,
	alpha : double,
	rectX : rect1d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dscal(N, alpha, rawX.ptr, rawX.offset)
end

terra sgemv_cpu_terra(
	layout : int,
	TransA : int,
	M : int,
	N : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_sgemv(layout, TransA, M, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra sgbmv_cpu_terra(
	layout : int,
	TransA : int,
	M : int,
	N : int,
	KL : int,
	KU : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_sgbmv(layout, TransA, M, N, KL, KU, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra strmv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_strmv(layout, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stbmv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	K : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_stbmv(layout, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stpmv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectAP : rect2d,
	rectX : rect1d,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawAP : float_ptr
	[get_raw_ptr_factory(2, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_stpmv(layout, Uplo, TransA, Diag, N, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra strsv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_strsv(layout, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stbsv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	K : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_stbsv(layout, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stpsv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectAP : rect2d,
	rectX : rect1d,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawAP : float_ptr
	[get_raw_ptr_factory(2, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_stpsv(layout, Uplo, TransA, Diag, N, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra dgemv_cpu_terra(
	layout : int,
	TransA : int,
	M : int,
	N : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dgemv(layout, TransA, M, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dgbmv_cpu_terra(
	layout : int,
	TransA : int,
	M : int,
	N : int,
	KL : int,
	KU : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dgbmv(layout, TransA, M, N, KL, KU, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dtrmv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtrmv(layout, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtbmv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	K : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtbmv(layout, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtpmv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectAP : rect2d,
	rectX : rect1d,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawAP : double_ptr
	[get_raw_ptr_factory(2, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtpmv(layout, Uplo, TransA, Diag, N, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra dtrsv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtrsv(layout, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtbsv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	K : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtbsv(layout, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtpsv_cpu_terra(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectAP : rect2d,
	rectX : rect1d,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t)

	var rawAP : double_ptr
	[get_raw_ptr_factory(2, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtpsv(layout, Uplo, TransA, Diag, N, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra ssymv_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_ssymv(layout, Uplo, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra ssbmv_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
	K : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_ssbmv(layout, Uplo, N, K, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra sspmv_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
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

	var rawAP : float_ptr
	[get_raw_ptr_factory(2, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_sspmv(layout, Uplo, N, alpha, rawAP.ptr, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra sger_cpu_terra(
	layout : int,
	M : int,
	N : int,
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

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	cblas.cblas_sger(layout, M, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra ssyr_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
	alpha : float,
	rectX : rect1d,
	rectA : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	cblas.cblas_ssyr(layout, Uplo, N, alpha, rawX.ptr, rawX.offset, rawA.ptr, rawA.offset)
end

terra sspr_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
	alpha : float,
	rectX : rect1d,
	rectAP : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawAP : float_ptr
	[get_raw_ptr_factory(2, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	cblas.cblas_sspr(layout, Uplo, N, alpha, rawX.ptr, rawX.offset, rawAP.ptr)
end

terra ssyr2_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
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

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	cblas.cblas_ssyr2(layout, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra sspr2_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
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

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	cblas.cblas_sspr2(layout, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr)
end

terra dsymv_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dsymv(layout, Uplo, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dsbmv_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
	K : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dsbmv(layout, Uplo, N, K, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dspmv_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
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

	var rawAP : double_ptr
	[get_raw_ptr_factory(2, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dspmv(layout, Uplo, N, alpha, rawAP.ptr, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dger_cpu_terra(
	layout : int,
	M : int,
	N : int,
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

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	cblas.cblas_dger(layout, M, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra dsyr_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
	alpha : double,
	rectX : rect1d,
	rectA : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	cblas.cblas_dsyr(layout, Uplo, N, alpha, rawX.ptr, rawX.offset, rawA.ptr, rawA.offset)
end

terra dspr_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
	alpha : double,
	rectX : rect1d,
	rectAP : rect2d,
	prX : clib.legion_physical_region_t,
	fldX : clib.legion_field_id_t,
	prAP : clib.legion_physical_region_t,
	fldAP : clib.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawAP : double_ptr
	[get_raw_ptr_factory(2, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	cblas.cblas_dspr(layout, Uplo, N, alpha, rawX.ptr, rawX.offset, rawAP.ptr)
end

terra dsyr2_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
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

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	cblas.cblas_dsyr2(layout, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra dspr2_cpu_terra(
	layout : int,
	Uplo : int,
	N : int,
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

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	cblas.cblas_dspr2(layout, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr)
end

terra sgemm_cpu_terra(
	layout : int,
	TransA : int,
	TransB : int,
	M : int,
	N : int,
	K : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	cblas.cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra ssymm_cpu_terra(
	layout : int,
	Side : int,
	Uplo : int,
	M : int,
	N : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	cblas.cblas_ssymm(layout, Side, Uplo, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra ssyrk_cpu_terra(
	layout : int,
	Uplo : int,
	Trans : int,
	N : int,
	K : int,
	alpha : float,
	rectA : rect2d,
	beta : float,
	rectC : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	cblas.cblas_ssyrk(layout, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, beta, rawC.ptr, rawC.offset)
end

terra ssyr2k_cpu_terra(
	layout : int,
	Uplo : int,
	Trans : int,
	N : int,
	K : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	cblas.cblas_ssyr2k(layout, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra strmm_cpu_terra(
	layout : int,
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	M : int,
	N : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	cblas.cblas_strmm(layout, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra strsm_cpu_terra(
	layout : int,
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	M : int,
	N : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	cblas.cblas_strsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra dgemm_cpu_terra(
	layout : int,
	TransA : int,
	TransB : int,
	M : int,
	N : int,
	K : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	cblas.cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra dsymm_cpu_terra(
	layout : int,
	Side : int,
	Uplo : int,
	M : int,
	N : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	cblas.cblas_dsymm(layout, Side, Uplo, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra dsyrk_cpu_terra(
	layout : int,
	Uplo : int,
	Trans : int,
	N : int,
	K : int,
	alpha : double,
	rectA : rect2d,
	beta : double,
	rectC : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prC : clib.legion_physical_region_t,
	fldC : clib.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	cblas.cblas_dsyrk(layout, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, beta, rawC.ptr, rawC.offset)
end

terra dsyr2k_cpu_terra(
	layout : int,
	Uplo : int,
	Trans : int,
	N : int,
	K : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	cblas.cblas_dsyr2k(layout, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra dtrmm_cpu_terra(
	layout : int,
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	M : int,
	N : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	cblas.cblas_dtrmm(layout, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra dtrsm_cpu_terra(
	layout : int,
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	M : int,
	N : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	prA : clib.legion_physical_region_t,
	fldA : clib.legion_field_id_t,
	prB : clib.legion_physical_region_t,
	fldB : clib.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	cblas.cblas_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

local tasks_h = "cblas_tasks.h"
local tasks_so = "cblas_tasks.so"
regentlib.save_tasks(tasks_h, tasks_so, nil, nil, nil, nil, false)
