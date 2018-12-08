
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

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
end

local raw_ptr = raw_ptr_factory(double)

terra get_raw_ptr_2d(rect: rect2d,
                     pr : c.legion_physical_region_t,
                     fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra get_raw_ptr_1d(rect: rect1d,
                     pr : c.legion_physical_region_t,
                     fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_1d(pr, fld)
  var subrect : c.legion_rect_1d_t
  var offsets : c.legion_byte_offset_t[1]
  var ptr = c.legion_accessor_array_1d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[0].offset / sizeof(double) }
end

terra sdsdot_terra(
	N : int
	alpha : float
	rectX : rect1d
	rectY : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.sdsdot(N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dsdot_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.dsdot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra sdot_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.sdot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra ddot_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.ddot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra snrm2_terra(
	N : int
	rectX : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.snrm2(N, rawX.ptr, rawX.offset)
end

terra sasum_terra(
	N : int
	rectX : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.sasum(N, rawX.ptr, rawX.offset)
end

terra dnrm2_terra(
	N : int
	rectX : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.dnrm2(N, rawX.ptr, rawX.offset)
end

terra dasum_terra(
	N : int
	rectX : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.dasum(N, rawX.ptr, rawX.offset)
end

terra sswap_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.sswap(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra scopy_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.scopy(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra saxpy_terra(
	N : int
	alpha : float
	rectX : rect1d
	rectY : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.saxpy(N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dswap_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.dswap(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra dcopy_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.dcopy(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra daxpy_terra(
	N : int
	alpha : double
	rectX : rect1d
	rectY : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.daxpy(N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset)
end

terra srot_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	c : float
	s : float
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.srot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, c, s)
end

terra srotm_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	rectP : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t
	prP : c.legion_physical_region_t
	fldP : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	var rawP = get_raw_pointer_1d(rectP, prP, fldP)
	cblas.srotm(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawP.ptr)
end

terra drot_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	c : double
	s : double
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.drot(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, c, s)
end

terra drotm_terra(
	N : int
	rectX : rect1d
	rectY : rect1d
	rectP : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t
	prP : c.legion_physical_region_t
	fldP : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	var rawP = get_raw_pointer_1d(rectP, prP, fldP)
	cblas.drotm(N, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawP.ptr)
end

terra sscal_terra(
	N : int
	alpha : float
	rectX : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.sscal(N, alpha, rawX.ptr, rawX.offset)
end

terra dscal_terra(
	N : int
	alpha : double
	rectX : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.dscal(N, alpha, rawX.ptr, rawX.offset)
end

terra sgemv_terra(
	TransA : int
	M : int
	N : int
	alpha : float
	rectA : rect2d
	rectX : rect1d
	beta : float
	rectY : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.sgemv(cblas.CblasColMajor, TransA, M, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra sgbmv_terra(
	TransA : int
	M : int
	N : int
	KL : int
	KU : int
	alpha : float
	rectA : rect2d
	rectX : rect1d
	beta : float
	rectY : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.sgbmv(cblas.CblasColMajor, TransA, M, N, KL, KU, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra strmv_terra(
	Uplo : int
	TransA : int
	Diag : int
	N : int
	rectA : rect2d
	rectX : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.strmv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stbmv_terra(
	Uplo : int
	TransA : int
	Diag : int
	N : int
	K : int
	rectA : rect2d
	rectX : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.stbmv(cblas.CblasColMajor, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra strsv_terra(
	Uplo : int
	TransA : int
	Diag : int
	N : int
	rectA : rect2d
	rectX : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.strsv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra stbsv_terra(
	Uplo : int
	TransA : int
	Diag : int
	N : int
	K : int
	rectA : rect2d
	rectX : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.stbsv(cblas.CblasColMajor, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dgemv_terra(
	TransA : int
	M : int
	N : int
	alpha : double
	rectA : rect2d
	rectX : rect1d
	beta : double
	rectY : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.dgemv(cblas.CblasColMajor, TransA, M, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dgbmv_terra(
	TransA : int
	M : int
	N : int
	KL : int
	KU : int
	alpha : double
	rectA : rect2d
	rectX : rect1d
	beta : double
	rectY : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.dgbmv(cblas.CblasColMajor, TransA, M, N, KL, KU, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dtrmv_terra(
	Uplo : int
	TransA : int
	Diag : int
	N : int
	rectA : rect2d
	rectX : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.dtrmv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtbmv_terra(
	Uplo : int
	TransA : int
	Diag : int
	N : int
	K : int
	rectA : rect2d
	rectX : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.dtbmv(cblas.CblasColMajor, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtrsv_terra(
	Uplo : int
	TransA : int
	Diag : int
	N : int
	rectA : rect2d
	rectX : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.dtrsv(cblas.CblasColMajor, Uplo, TransA, Diag, N, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra dtbsv_terra(
	Uplo : int
	TransA : int
	Diag : int
	N : int
	K : int
	rectA : rect2d
	rectX : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	cblas.dtbsv(cblas.CblasColMajor, Uplo, TransA, Diag, N, K, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset)
end

terra ssymv_terra(
	Uplo : int
	N : int
	alpha : float
	rectA : rect2d
	rectX : rect1d
	beta : float
	rectY : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.ssymv(cblas.CblasColMajor, Uplo, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra ssbmv_terra(
	Uplo : int
	N : int
	K : int
	alpha : float
	rectA : rect2d
	rectX : rect1d
	beta : float
	rectY : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.ssbmv(cblas.CblasColMajor, Uplo, N, K, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra sger_terra(
	M : int
	N : int
	alpha : float
	rectX : rect1d
	rectY : rect1d
	rectA : rect2d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	cblas.sger(cblas.CblasColMajor, M, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra ssyr_terra(
	Uplo : int
	N : int
	alpha : float
	rectX : rect1d
	rectA : rect2d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	cblas.ssyr(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawA.ptr, rawA.offset)
end

terra sspr_terra(
	Uplo : int
	N : int
	alpha : float
	rectX : rect1d
	rectAp : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prAp : c.legion_physical_region_t
	fldAp : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawAp = get_raw_pointer_1d(rectAp, prAp, fldAp)
	cblas.sspr(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawAp.ptr)
end

terra ssyr2_terra(
	Uplo : int
	N : int
	alpha : float
	rectX : rect1d
	rectY : rect1d
	rectA : rect2d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	cblas.ssyr2(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra sspr2_terra(
	Uplo : int
	N : int
	alpha : float
	rectX : rect1d
	rectY : rect1d
	rectA : rect2d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	cblas.sspr2(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr)
end

terra dsymv_terra(
	Uplo : int
	N : int
	alpha : double
	rectA : rect2d
	rectX : rect1d
	beta : double
	rectY : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.dsymv(cblas.CblasColMajor, Uplo, N, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dsbmv_terra(
	Uplo : int
	N : int
	K : int
	alpha : double
	rectA : rect2d
	rectX : rect1d
	beta : double
	rectY : rect1d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	cblas.dsbmv(cblas.CblasColMajor, Uplo, N, K, alpha, rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)
end

terra dger_terra(
	M : int
	N : int
	alpha : double
	rectX : rect1d
	rectY : rect1d
	rectA : rect2d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	cblas.dger(cblas.CblasColMajor, M, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra dsyr_terra(
	Uplo : int
	N : int
	alpha : double
	rectX : rect1d
	rectA : rect2d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	cblas.dsyr(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawA.ptr, rawA.offset)
end

terra dspr_terra(
	Uplo : int
	N : int
	alpha : double
	rectX : rect1d
	rectAp : rect1d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prAp : c.legion_physical_region_t
	fldAp : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawAp = get_raw_pointer_1d(rectAp, prAp, fldAp)
	cblas.dspr(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawAp.ptr)
end

terra dsyr2_terra(
	Uplo : int
	N : int
	alpha : double
	rectX : rect1d
	rectY : rect1d
	rectA : rect2d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	cblas.dsyr2(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr, rawA.offset)
end

terra dspr2_terra(
	Uplo : int
	N : int
	alpha : double
	rectX : rect1d
	rectY : rect1d
	rectA : rect2d
	prX : c.legion_physical_region_t
	fldX : c.legion_field_id_t
	prY : c.legion_physical_region_t
	fldY : c.legion_field_id_t
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t)

	var rawX = get_raw_pointer_1d(rectX, prX, fldX)
	var rawY = get_raw_pointer_1d(rectY, prY, fldY)
	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	cblas.dspr2(cblas.CblasColMajor, Uplo, N, alpha, rawX.ptr, rawX.offset, rawY.ptr, rawY.offset, rawA.ptr)
end

terra sgemm_terra(
	TransA : int
	TransB : int
	M : int
	N : int
	K : int
	alpha : float
	rectA : rect2d
	rectB : rect2d
	beta : float
	rectC : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prB : c.legion_physical_region_t
	fldB : c.legion_field_id_t
	prC : c.legion_physical_region_t
	fldC : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawB = get_raw_pointer_2d(rectB, prB, fldB)
	var rawC = get_raw_pointer_2d(rectC, prC, fldC)
	cblas.sgemm(cblas.CblasColMajor, TransA, TransB, M, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra ssymm_terra(
	Side : int
	Uplo : int
	M : int
	N : int
	alpha : float
	rectA : rect2d
	rectB : rect2d
	beta : float
	rectC : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prB : c.legion_physical_region_t
	fldB : c.legion_field_id_t
	prC : c.legion_physical_region_t
	fldC : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawB = get_raw_pointer_2d(rectB, prB, fldB)
	var rawC = get_raw_pointer_2d(rectC, prC, fldC)
	cblas.ssymm(cblas.CblasColMajor, Side, Uplo, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra ssyrk_terra(
	Uplo : int
	Trans : int
	N : int
	K : int
	alpha : float
	rectA : rect2d
	beta : float
	rectC : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prC : c.legion_physical_region_t
	fldC : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawC = get_raw_pointer_2d(rectC, prC, fldC)
	cblas.ssyrk(cblas.CblasColMajor, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, beta, rawC.ptr, rawC.offset)
end

terra ssyr2k_terra(
	Uplo : int
	Trans : int
	N : int
	K : int
	alpha : float
	rectA : rect2d
	rectB : rect2d
	beta : float
	rectC : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prB : c.legion_physical_region_t
	fldB : c.legion_field_id_t
	prC : c.legion_physical_region_t
	fldC : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawB = get_raw_pointer_2d(rectB, prB, fldB)
	var rawC = get_raw_pointer_2d(rectC, prC, fldC)
	cblas.ssyr2k(cblas.CblasColMajor, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra strmm_terra(
	Side : int
	Uplo : int
	TransA : int
	Diag : int
	M : int
	N : int
	alpha : float
	rectA : rect2d
	rectB : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prB : c.legion_physical_region_t
	fldB : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawB = get_raw_pointer_2d(rectB, prB, fldB)
	cblas.strmm(cblas.CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra strsm_terra(
	Side : int
	Uplo : int
	TransA : int
	Diag : int
	M : int
	N : int
	alpha : float
	rectA : rect2d
	rectB : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prB : c.legion_physical_region_t
	fldB : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawB = get_raw_pointer_2d(rectB, prB, fldB)
	cblas.strsm(cblas.CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra dgemm_terra(
	TransA : int
	TransB : int
	M : int
	N : int
	K : int
	alpha : double
	rectA : rect2d
	rectB : rect2d
	beta : double
	rectC : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prB : c.legion_physical_region_t
	fldB : c.legion_field_id_t
	prC : c.legion_physical_region_t
	fldC : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawB = get_raw_pointer_2d(rectB, prB, fldB)
	var rawC = get_raw_pointer_2d(rectC, prC, fldC)
	cblas.dgemm(cblas.CblasColMajor, TransA, TransB, M, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra dsymm_terra(
	Side : int
	Uplo : int
	M : int
	N : int
	alpha : double
	rectA : rect2d
	rectB : rect2d
	beta : double
	rectC : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prB : c.legion_physical_region_t
	fldB : c.legion_field_id_t
	prC : c.legion_physical_region_t
	fldC : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawB = get_raw_pointer_2d(rectB, prB, fldB)
	var rawC = get_raw_pointer_2d(rectC, prC, fldC)
	cblas.dsymm(cblas.CblasColMajor, Side, Uplo, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra dsyrk_terra(
	Uplo : int
	Trans : int
	N : int
	K : int
	alpha : double
	rectA : rect2d
	beta : double
	rectC : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prC : c.legion_physical_region_t
	fldC : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawC = get_raw_pointer_2d(rectC, prC, fldC)
	cblas.dsyrk(cblas.CblasColMajor, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, beta, rawC.ptr, rawC.offset)
end

terra dsyr2k_terra(
	Uplo : int
	Trans : int
	N : int
	K : int
	alpha : double
	rectA : rect2d
	rectB : rect2d
	beta : double
	rectC : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prB : c.legion_physical_region_t
	fldB : c.legion_field_id_t
	prC : c.legion_physical_region_t
	fldC : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawB = get_raw_pointer_2d(rectB, prB, fldB)
	var rawC = get_raw_pointer_2d(rectC, prC, fldC)
	cblas.dsyr2k(cblas.CblasColMajor, Uplo, Trans, N, K, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset, beta, rawC.ptr, rawC.offset)
end

terra dtrmm_terra(
	Side : int
	Uplo : int
	TransA : int
	Diag : int
	M : int
	N : int
	alpha : double
	rectA : rect2d
	rectB : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prB : c.legion_physical_region_t
	fldB : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawB = get_raw_pointer_2d(rectB, prB, fldB)
	cblas.dtrmm(cblas.CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

terra dtrsm_terra(
	Side : int
	Uplo : int
	TransA : int
	Diag : int
	M : int
	N : int
	alpha : double
	rectA : rect2d
	rectB : rect2d
	prA : c.legion_physical_region_t
	fldA : c.legion_field_id_t
	prB : c.legion_physical_region_t
	fldB : c.legion_field_id_t)

	var rawA = get_raw_pointer_2d(rectA, prA, fldA)
	var rawB = get_raw_pointer_2d(rectB, prB, fldB)
	cblas.dtrsm(cblas.CblasColMajor, Side, Uplo, TransA, Diag, M, N, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

