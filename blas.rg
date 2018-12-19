
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

terra sdsdot_terra(
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
	cblas.cblas_sdsdot(N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dsdot_terra(
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
	cblas.cblas_dsdot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
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
	cblas.cblas_sdot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
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
	cblas.cblas_ddot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra snrm2_terra(
	N : int,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_snrm2(N, rawX.ptr, rawX.offset)
end

terra sasum_terra(
	N : int,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	cblas.cblas_sasum(N, rawX.ptr, rawX.offset)
end

terra dnrm2_terra(
	N : int,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dnrm2(N, rawX.ptr, rawX.offset)
end

terra dasum_terra(
	N : int,
	rectX : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	cblas.cblas_dasum(N, rawX.ptr, rawX.offset)
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

terra srot_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	c : float,
	s : float,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	cblas.cblas_srot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, c, s)
end

terra srotm_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	rectP : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t,
	prP : c.legion_physical_region_t,
	fldP : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawY : float_ptr
	[get_raw_ptr_factory(1, float, rectY, prY, fldY, rawY, float_ptr)]
	var rawP : float_ptr
	[get_raw_ptr_factory(1, float, rectP, prP, fldP, rawP, float_ptr)]
	cblas.cblas_srotm(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawP.ptr)
end

terra drot_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	c : double,
	s : double,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	cblas.cblas_drot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, c, s)
end

terra drotm_terra(
	N : int,
	rectX : rect1d,
	rectY : rect1d,
	rectP : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prY : c.legion_physical_region_t,
	fldY : c.legion_field_id_t,
	prP : c.legion_physical_region_t,
	fldP : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawY : double_ptr
	[get_raw_ptr_factory(1, double, rectY, prY, fldY, rawY, double_ptr)]
	var rawP : double_ptr
	[get_raw_ptr_factory(1, double, rectP, prP, fldP, rawP, double_ptr)]
	cblas.cblas_drotm(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawP.ptr)
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
	rectAp : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prAp : c.legion_physical_region_t,
	fldAp : c.legion_field_id_t)

	var rawX : float_ptr
	[get_raw_ptr_factory(1, float, rectX, prX, fldX, rawX, float_ptr)]
	var rawAp : float_ptr
	[get_raw_ptr_factory(1, float, rectAp, prAp, fldAp, rawAp, float_ptr)]
	cblas.cblas_sspr(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawAp.ptr)
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
	rectAp : rect1d,
	prX : c.legion_physical_region_t,
	fldX : c.legion_field_id_t,
	prAp : c.legion_physical_region_t,
	fldAp : c.legion_field_id_t)

	var rawX : double_ptr
	[get_raw_ptr_factory(1, double, rectX, prX, fldX, rawX, double_ptr)]
	var rawAp : double_ptr
	[get_raw_ptr_factory(1, double, rectAp, prAp, fldAp, rawAp, double_ptr)]
	cblas.cblas_dspr(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawAp.ptr)
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

