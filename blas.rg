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

terralib.linklibrary("libcblas.so")
local cblas = terralib.includec("cblas.h")

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

terra sdot_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	return cblas.cblas_sdot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra ddot_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	return cblas.cblas_ddot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra snrm2_terra(
	N : int,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	return cblas.cblas_snrm2(N, rawX.ptr, rawX.offset)
end

terra sasum_terra(
	N : int,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	return cblas.cblas_sasum(N, rawX.ptr, rawX.offset)
end

terra dnrm2_terra(
	N : int,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	return cblas.cblas_dnrm2(N, rawX.ptr, rawX.offset)
end

terra dasum_terra(
	N : int,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	return cblas.cblas_dasum(N, rawX.ptr, rawX.offset)
end

terra isamax_terra(
	N : int,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	return cblas.cblas_isamax(N, rawX.ptr, rawX.offset)
end

terra idamax_terra(
	N : int,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	return cblas.cblas_idamax(N, rawX.ptr, rawX.offset)
end

terra sswap_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_sswap(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra scopy_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_scopy(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra saxpy_terra(
	N : int,
	alpha : float,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_saxpy(N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dswap_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dswap(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dcopy_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dcopy(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra daxpy_terra(
	N : int,
	alpha : double,
	rectX : rect1d,
	rectY : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_daxpy(N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra sscal_terra(
	N : int,
	alpha : float,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_sscal(N, alpha, rawX.ptr, rawX.offset)
end

terra dscal_terra(
	N : int,
	alpha : double,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dscal(N, alpha, rawX.ptr, rawX.offset)
end

terra sgemv_terra(
	TransA : int,
	M : int,
	N : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_sgemv(cblas.CblasColMajor, TransA, M, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra sgbmv_terra(
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
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_sgbmv(cblas.CblasColMajor, TransA, M, N, KL, KU, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra strmv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_strmv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stbmv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	K : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_stbmv(cblas.CblasColMajor, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stpmv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectAP : rect1d,
	rectX : rect1d,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawAP : float_ptr
	[get_raw_ptr_factory(1, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_stpmv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra strsv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_strsv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stbsv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	K : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_stbsv(cblas.CblasColMajor, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stpsv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectAP : rect1d,
	rectX : rect1d,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawAP : float_ptr
	[get_raw_ptr_factory(1, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_stpsv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra dgemv_terra(
	TransA : int,
	M : int,
	N : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dgemv(cblas.CblasColMajor, TransA, M, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dgbmv_terra(
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
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dgbmv(cblas.CblasColMajor, TransA, M, N, KL, KU, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dtrmv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtrmv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtbmv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	K : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtbmv(cblas.CblasColMajor, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtpmv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectAP : rect1d,
	rectX : rect1d,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawAP : double_ptr
	[get_raw_ptr_factory(1, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtpmv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra dtrsv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtrsv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtbsv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	K : int,
	rectA : rect2d,
	rectX : rect1d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtbsv(cblas.CblasColMajor, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtpsv_terra(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	rectAP : rect1d,
	rectX : rect1d,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawAP : double_ptr
	[get_raw_ptr_factory(1, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dtpsv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawAP.ptr, rawX.ptr, rawX.offset)
end

terra ssymv_terra(
	Uplo : int,
	N : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_ssymv(cblas.CblasColMajor, Uplo, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra ssbmv_terra(
	Uplo : int,
	N : int,
	K : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_ssbmv(cblas.CblasColMajor, Uplo, N, K, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra sspmv_terra(
	Uplo : int,
	N : int,
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

	var rawAP : float_ptr
	[get_raw_ptr_factory(1, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_sspmv(cblas.CblasColMajor, Uplo, N, alpha, rawAP.ptr, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra sger_terra(
	M : int,
	N : int,
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

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	cblas.cblas_sger(cblas.CblasColMajor, M, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra ssyr_terra(
	Uplo : int,
	N : int,
	alpha : float,
	rectX : rect1d,
	rectA : rect2d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	cblas.cblas_ssyr(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawA.ptr, rawA.offset)
end

terra sspr_terra(
	Uplo : int,
	N : int,
	alpha : float,
	rectX : rect1d,
	rectAP : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawAP : float_ptr
	[get_raw_ptr_factory(1, float, rectAP, prAP, fldAP, rawAP, float_ptr)]
	cblas.cblas_sspr(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawAP.ptr)
end

terra ssyr2_terra(
	Uplo : int,
	N : int,
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

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	cblas.cblas_ssyr2(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra sspr2_terra(
	Uplo : int,
	N : int,
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

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	cblas.cblas_sspr2(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr)
end

terra dsymv_terra(
	Uplo : int,
	N : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dsymv(cblas.CblasColMajor, Uplo, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dsbmv_terra(
	Uplo : int,
	N : int,
	K : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dsbmv(cblas.CblasColMajor, Uplo, N, K, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dspmv_terra(
	Uplo : int,
	N : int,
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

	var rawAP : double_ptr
	[get_raw_ptr_factory(1, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_dspmv(cblas.CblasColMajor, Uplo, N, alpha, rawAP.ptr, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dger_terra(
	M : int,
	N : int,
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

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	cblas.cblas_dger(cblas.CblasColMajor, M, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra dsyr_terra(
	Uplo : int,
	N : int,
	alpha : double,
	rectX : rect1d,
	rectA : rect2d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	cblas.cblas_dsyr(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawA.ptr, rawA.offset)
end

terra dspr_terra(
	Uplo : int,
	N : int,
	alpha : double,
	rectX : rect1d,
	rectAP : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prAP : c.legion_physical_region_t,
	fldAP : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawAP : double_ptr
	[get_raw_ptr_factory(1, double, rectAP, prAP, fldAP, rawAP, double_ptr)]
	cblas.cblas_dspr(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawAP.ptr)
end

terra dsyr2_terra(
	Uplo : int,
	N : int,
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

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	cblas.cblas_dsyr2(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra dspr2_terra(
	Uplo : int,
	N : int,
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

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	cblas.cblas_dspr2(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr)
end

terra sgemm_terra(
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
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	cblas.cblas_sgemm(cblas.CblasColMajor, TransA, TransB, M, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra ssymm_terra(
	Side : int,
	Uplo : int,
	M : int,
	N : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	cblas.cblas_ssymm(cblas.CblasColMajor, Side, Uplo, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra ssyrk_terra(
	Uplo : int,
	Trans : int,
	N : int,
	K : int,
	alpha : float,
	rectA : rect2d,
	beta : float,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	cblas.cblas_ssyrk(cblas.CblasColMajor, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, beta, rawC.ptr, rawC.offset)
end

terra ssyr2k_terra(
	Uplo : int,
	Trans : int,
	N : int,
	K : int,
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

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	var rawC : float_ptr
	[get_raw_ptr_factory(2, float, rectC, prC, fldC, rawC, float_ptr)]
	cblas.cblas_ssyr2k(cblas.CblasColMajor, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra strmm_terra(
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	M : int,
	N : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	cblas.cblas_strmm(cblas.CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra strsm_terra(
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	M : int,
	N : int,
	alpha : float,
	rectA : rect2d,
	rectB : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t)

	var rawA : float_ptr
	[get_raw_ptr_factory(2, float, rectA, prA, fldA, rawA, float_ptr)]
	var rawB : float_ptr
	[get_raw_ptr_factory(2, float, rectB, prB, fldB, rawB, float_ptr)]
	cblas.cblas_strsm(cblas.CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra dgemm_terra(
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
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	cblas.cblas_dgemm(cblas.CblasColMajor, TransA, TransB, M, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra dsymm_terra(
	Side : int,
	Uplo : int,
	M : int,
	N : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	cblas.cblas_dsymm(cblas.CblasColMajor, Side, Uplo, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra dsyrk_terra(
	Uplo : int,
	Trans : int,
	N : int,
	K : int,
	alpha : double,
	rectA : rect2d,
	beta : double,
	rectC : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prC : c.legion_physical_region_t,
	fldC : c.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	cblas.cblas_dsyrk(cblas.CblasColMajor, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, beta, rawC.ptr, rawC.offset)
end

terra dsyr2k_terra(
	Uplo : int,
	Trans : int,
	N : int,
	K : int,
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

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	var rawC : double_ptr
	[get_raw_ptr_factory(2, double, rectC, prC, fldC, rawC, double_ptr)]
	cblas.cblas_dsyr2k(cblas.CblasColMajor, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra dtrmm_terra(
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	M : int,
	N : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	cblas.cblas_dtrmm(cblas.CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra dtrsm_terra(
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	M : int,
	N : int,
	alpha : double,
	rectA : rect2d,
	rectB : rect2d,
	prA : c.legion_physical_region_t,
	fldA : c.legion_field_id_t,
	prB : c.legion_physical_region_t,
	fldB : c.legion_field_id_t)

	var rawA : double_ptr
	[get_raw_ptr_factory(2, double, rectA, prA, fldA, rawA, double_ptr)]
	var rawB : double_ptr
	[get_raw_ptr_factory(2, double, rectB, prB, fldB, rawB, double_ptr)]
	cblas.cblas_dtrsm(cblas.CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end


__demand(__leaf)
task sdot(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads(X),
	reads(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = (sizeX-0)/1
	return sdot_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task ddot(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads(X),
	reads(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = (sizeX-0)/1
	return ddot_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task snrm2(
	X : region(ispace(int1d), float))
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = (sizeX-0)/1
	return snrm2_terra(N, rectX, __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task sasum(
	X : region(ispace(int1d), float))
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = (sizeX-0)/1
	return sasum_terra(N, rectX, __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task dnrm2(
	X : region(ispace(int1d), double))
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = (sizeX-0)/1
	return dnrm2_terra(N, rectX, __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task dasum(
	X : region(ispace(int1d), double))
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = (sizeX-0)/1
	return dasum_terra(N, rectX, __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task isamax(
	X : region(ispace(int1d), float))
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = (sizeX-0)/1
	return isamax_terra(N, rectX, __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task idamax(
	X : region(ispace(int1d), double))
where
	reads(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = (sizeX-0)/1
	return idamax_terra(N, rectX, __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task sswap(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads writes(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = (sizeX-0)/1
	sswap_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task scopy(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = (sizeX-0)/1
	scopy_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task saxpy(
	alpha : float,
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = (sizeX-0)/1
	saxpy_terra(N, alpha, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task dswap(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads writes(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = (sizeX-0)/1
	dswap_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task dcopy(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = (sizeX-0)/1
	dcopy_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task daxpy(
	alpha : double,
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads(X),
	reads writes(Y)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = (sizeX-0)/1
	daxpy_terra(N, alpha, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task sscal(
	alpha : float,
	X : region(ispace(int1d), float))
where
	reads writes(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = (sizeX-0)/1
	sscal_terra(N, alpha, rectX, __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task dscal(
	alpha : double,
	X : region(ispace(int1d), double))
where
	reads writes(X)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = (sizeX-0)/1
	dscal_terra(N, alpha, rectX, __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task sgemv(
	TransA : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var M = sizeA.x
	var N = sizeA.y
	sgemv_terra(TransA, M, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task sgbmv(
	TransA : int,
	M : int,
	N : int,
	KL : int,
	KU : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	sgbmv_terra(TransA, M, N, KL, KU, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task strmv(
	Uplo : int,
	TransA : int,
	Diag : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = sizeA.x
	strmv_terra(Uplo, TransA, Diag, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task stbmv(
	Uplo : int,
	TransA : int,
	Diag : int,
	K : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = sizeA.y
	stbmv_terra(Uplo, TransA, Diag, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task stpmv(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	AP : region(ispace(int1d), float),
	X : region(ispace(int1d), float))
where
	reads(AP),
	reads writes(X)
do
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	stpmv_terra(Uplo, TransA, Diag, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task strsv(
	Uplo : int,
	TransA : int,
	Diag : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = sizeA.x
	strsv_terra(Uplo, TransA, Diag, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task stbsv(
	Uplo : int,
	TransA : int,
	Diag : int,
	K : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = sizeA.y
	stbsv_terra(Uplo, TransA, Diag, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task stpsv(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	AP : region(ispace(int1d), float),
	X : region(ispace(int1d), float))
where
	reads(AP),
	reads writes(X)
do
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	stpsv_terra(Uplo, TransA, Diag, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task dgemv(
	TransA : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var M = sizeA.x
	var N = sizeA.y
	dgemv_terra(TransA, M, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task dgbmv(
	TransA : int,
	M : int,
	N : int,
	KL : int,
	KU : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	dgbmv_terra(TransA, M, N, KL, KU, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task dtrmv(
	Uplo : int,
	TransA : int,
	Diag : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = sizeA.x
	dtrmv_terra(Uplo, TransA, Diag, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task dtbmv(
	Uplo : int,
	TransA : int,
	Diag : int,
	K : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = sizeA.y
	dtbmv_terra(Uplo, TransA, Diag, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task dtpmv(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	AP : region(ispace(int1d), double),
	X : region(ispace(int1d), double))
where
	reads(AP),
	reads writes(X)
do
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	dtpmv_terra(Uplo, TransA, Diag, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task dtrsv(
	Uplo : int,
	TransA : int,
	Diag : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = sizeA.x
	dtrsv_terra(Uplo, TransA, Diag, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task dtbsv(
	Uplo : int,
	TransA : int,
	Diag : int,
	K : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
do
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var N = sizeA.y
	dtbsv_terra(Uplo, TransA, Diag, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task dtpsv(
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	AP : region(ispace(int1d), double),
	X : region(ispace(int1d), double))
where
	reads(AP),
	reads writes(X)
do
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	dtpsv_terra(Uplo, TransA, Diag, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
end

__demand(__leaf)
task ssymv(
	Uplo : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = sizeA.x
	ssymv_terra(Uplo, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task ssbmv(
	Uplo : int,
	K : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = sizeA.y
	ssbmv_terra(Uplo, N, K, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task sspmv(
	Uplo : int,
	N : int,
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
	var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	sspmv_terra(Uplo, N, alpha, rectAP, rectX, beta, rectY, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var M = sizeX
	var N = sizeY
	sger_terra(M, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__leaf)
task ssyr(
	Uplo : int,
	alpha : float,
	X : region(ispace(int1d), float),
	A : region(ispace(int2d), float))
where
	reads(X),
	reads writes(A)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var N = (sizeX-1-0)/1+1
	ssyr_terra(Uplo, N, alpha, rectX, rectA, __physical(X)[0], __fields(X)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__leaf)
task sspr(
	Uplo : int,
	N : int,
	alpha : float,
	X : region(ispace(int1d), float),
	AP : region(ispace(int1d), float))
where
	reads(X),
	reads writes(AP)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
	sspr_terra(Uplo, N, alpha, rectX, rectAP, __physical(X)[0], __fields(X)[0], __physical(AP)[0], __fields(AP)[0])
end

__demand(__leaf)
task ssyr2(
	Uplo : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var N = 0
	if [bool]((sizeX-1-0)/1+1 <=(sizeY-1-0)/1+1) then
		N = (sizeX-1-0)/1+1
	else
		N = (sizeY-1-0)/1+1
	end

	ssyr2_terra(Uplo, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__leaf)
task sspr2(
	Uplo : int,
	N : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	sspr2_terra(Uplo, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__leaf)
task dsymv(
	Uplo : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = sizeA.x
	dsymv_terra(Uplo, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task dsbmv(
	Uplo : int,
	K : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var N = sizeA.y
	dsbmv_terra(Uplo, N, K, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
task dspmv(
	Uplo : int,
	N : int,
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
	var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	dspmv_terra(Uplo, N, alpha, rectAP, rectX, beta, rectY, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
end

__demand(__leaf)
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var M = sizeX
	var N = sizeY
	dger_terra(M, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__leaf)
task dsyr(
	Uplo : int,
	alpha : double,
	X : region(ispace(int1d), double),
	A : region(ispace(int2d), double))
where
	reads(X),
	reads writes(A)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var N = (sizeX-1-0)/1+1
	dsyr_terra(Uplo, N, alpha, rectX, rectA, __physical(X)[0], __fields(X)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__leaf)
task dspr(
	Uplo : int,
	N : int,
	alpha : double,
	X : region(ispace(int1d), double),
	AP : region(ispace(int1d), double))
where
	reads(X),
	reads writes(AP)
do
	var rectX = X.bounds
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectAP = AP.bounds
	var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
	dspr_terra(Uplo, N, alpha, rectX, rectAP, __physical(X)[0], __fields(X)[0], __physical(AP)[0], __fields(AP)[0])
end

__demand(__leaf)
task dsyr2(
	Uplo : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	var N = 0
	if [bool]((sizeX-1-0)/1+1 <=(sizeY-1-0)/1+1) then
		N = (sizeX-1-0)/1+1
	else
		N = (sizeY-1-0)/1+1
	end

	dsyr2_terra(Uplo, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__leaf)
task dspr2(
	Uplo : int,
	N : int,
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
	var sizeX = rectX.hi - rectX.lo + {1, 1}
	var rectY = Y.bounds
	var sizeY = rectY.hi - rectY.lo + {1, 1}
	var rectA = A.bounds
	var sizeA = rectA.hi - rectA.lo + {1, 1}
	dspr2_terra(Uplo, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
end

__demand(__leaf)
task sgemm(
	TransA : int,
	TransB : int,
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
	var M = 0
	if [bool](TransA) then
		M = sizeA.y
	else
		M = sizeA.x
	end

	var N = 0
	if [bool](TransB) then
		N = sizeB.x
	else
		N = sizeB.y
	end

	var K = 0
	if [bool](TransA) then
		K = sizeA.x
	else
		K = sizeA.y
	end

	sgemm_terra(TransA, TransB, M, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__leaf)
task ssymm(
	Side : int,
	Uplo : int,
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
	var M = 0
	if [bool](Side) then
		M = sizeB.x
	else
		M = sizeA.x
	end

	var N = 0
	if [bool](Side) then
		N = sizeA.y
	else
		N = sizeB.y
	end

	ssymm_terra(Side, Uplo, M, N, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__leaf)
task ssyrk(
	Uplo : int,
	Trans : int,
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
	var N = 0
	if [bool](Trans) then
		N = sizeA.y
	else
		N = sizeA.x
	end

	var K = 0
	if [bool](Trans) then
		K = sizeA.x
	else
		K = sizeA.y
	end

	ssyrk_terra(Uplo, Trans, N, K, alpha, rectA, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__leaf)
task ssyr2k(
	Uplo : int,
	Trans : int,
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
	var N = 0
	if [bool](Trans) then
		N = sizeA.y
	else
		N = sizeA.x
	end

	var K = 0
	if [bool](Trans) then
		K = sizeA.x
	else
		K = sizeA.y
	end

	ssyr2k_terra(Uplo, Trans, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__leaf)
task strmm(
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
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
	var M = 0
	if [bool](Side) then
		M = sizeB.y
	else
		M = sizeA.y
	end

	var N = sizeB.y
	strmm_terra(Side, Uplo, TransA, Diag, M, N, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
end

__demand(__leaf)
task strsm(
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
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
	var M = sizeB.x
	var N = sizeB.y
	strsm_terra(Side, Uplo, TransA, Diag, M, N, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
end

__demand(__leaf)
task dgemm(
	TransA : int,
	TransB : int,
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
	var M = 0
	if [bool](TransA) then
		M = sizeA.y
	else
		M = sizeA.x
	end

	var N = 0
	if [bool](TransB) then
		N = sizeB.x
	else
		N = sizeB.y
	end

	var K = 0
	if [bool](TransA) then
		K = sizeA.x
	else
		K = sizeA.y
	end

	dgemm_terra(TransA, TransB, M, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__leaf)
task dsymm(
	Side : int,
	Uplo : int,
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
	var M = 0
	if [bool](Side) then
		M = sizeB.x
	else
		M = sizeA.x
	end

	var N = 0
	if [bool](Side) then
		N = sizeA.y
	else
		N = sizeB.y
	end

	dsymm_terra(Side, Uplo, M, N, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__leaf)
task dsyrk(
	Uplo : int,
	Trans : int,
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
	var N = 0
	if [bool](Trans) then
		N = sizeA.y
	else
		N = sizeA.x
	end

	var K = 0
	if [bool](Trans) then
		K = sizeA.x
	else
		K = sizeA.y
	end

	dsyrk_terra(Uplo, Trans, N, K, alpha, rectA, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__leaf)
task dsyr2k(
	Uplo : int,
	Trans : int,
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
	var N = 0
	if [bool](Trans) then
		N = sizeA.y
	else
		N = sizeA.x
	end

	var K = 0
	if [bool](Trans) then
		K = sizeA.x
	else
		K = sizeA.y
	end

	dsyr2k_terra(Uplo, Trans, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
end

__demand(__leaf)
task dtrmm(
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
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
	var M = 0
	if [bool](Side) then
		M = sizeB.y
	else
		M = sizeA.y
	end

	var N = sizeB.y
	dtrmm_terra(Side, Uplo, TransA, Diag, M, N, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
end

__demand(__leaf)
task dtrsm(
	Side : int,
	Uplo : int,
	TransA : int,
	Diag : int,
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
	var M = sizeB.x
	var N = sizeB.y
	dtrsm_terra(Side, Uplo, TransA, Diag, M, N, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
end
