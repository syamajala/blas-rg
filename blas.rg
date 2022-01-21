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

extern task sdot(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads(X),
	reads(Y)
end

extern task ddot(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads(X),
	reads(Y)
end

extern task snrm2(
	X : region(ispace(int1d), float))
where
	reads(X)
end

extern task sasum(
	X : region(ispace(int1d), float))
where
	reads(X)
end

extern task dnrm2(
	X : region(ispace(int1d), double))
where
	reads(X)
end

extern task dasum(
	X : region(ispace(int1d), double))
where
	reads(X)
end

extern task isamax(
	X : region(ispace(int1d), float))
where
	reads(X)
end

extern task idamax(
	X : region(ispace(int1d), double))
where
	reads(X)
end

extern task sswap(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads writes(X),
	reads writes(Y)
end

extern task scopy(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads(X),
	reads writes(Y)
end

extern task saxpy(
	alpha : float,
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float))
where
	reads(X),
	reads writes(Y)
end

extern task dswap(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads writes(X),
	reads writes(Y)
end

extern task dcopy(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads(X),
	reads writes(Y)
end

extern task daxpy(
	alpha : double,
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double))
where
	reads(X),
	reads writes(Y)
end

extern task srotg(
	a : float,
	b : float,
	c : float,
	s : float)

extern task srotmg(
	d1 : float,
	d2 : float,
	b1 : float,
	b2 : float,
	P : region(ispace(int1d), float))
where
	reads writes(P)
end

extern task srot(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float),
	c : float,
	s : float)
where
	reads writes(X),
	reads writes(Y)
end

extern task srotm(
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float),
	P : region(ispace(int1d), float))
where
	reads writes(X),
	reads writes(Y),
	reads writes(P)
end

extern task drotg(
	a : double,
	b : double,
	c : double,
	s : double)

extern task drotmg(
	d1 : double,
	d2 : double,
	b1 : double,
	b2 : double,
	P : region(ispace(int1d), double))
where
	reads writes(P)
end

extern task drot(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double),
	c : double,
	s : double)
where
	reads writes(X),
	reads writes(Y)
end

extern task drotm(
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double),
	P : region(ispace(int1d), double))
where
	reads writes(X),
	reads writes(Y),
	reads writes(P)
end

extern task sscal(
	alpha : float,
	X : region(ispace(int1d), float))
where
	reads writes(X)
end

extern task dscal(
	alpha : double,
	X : region(ispace(int1d), double))
where
	reads writes(X)
end

extern task sgemv(
	layout : int,
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
end

extern task sgbmv(
	layout : int,
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
end

extern task strmv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
end

extern task stbmv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	K : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
end

extern task stpmv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	AP : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(AP),
	reads writes(X)
end

extern task strsv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
end

extern task stbsv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	K : int,
	A : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(A),
	reads writes(X)
end

extern task stpsv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	AP : region(ispace(int2d), float),
	X : region(ispace(int1d), float))
where
	reads(AP),
	reads writes(X)
end

extern task dgemv(
	layout : int,
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
end

extern task dgbmv(
	layout : int,
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
end

extern task dtrmv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
end

extern task dtbmv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	K : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
end

extern task dtpmv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	AP : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(AP),
	reads writes(X)
end

extern task dtrsv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
end

extern task dtbsv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	K : int,
	A : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(A),
	reads writes(X)
end

extern task dtpsv(
	layout : int,
	Uplo : int,
	TransA : int,
	Diag : int,
	N : int,
	AP : region(ispace(int2d), double),
	X : region(ispace(int1d), double))
where
	reads(AP),
	reads writes(X)
end

extern task ssymv(
	layout : int,
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
end

extern task ssbmv(
	layout : int,
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
end

extern task sspmv(
	layout : int,
	Uplo : int,
	N : int,
	alpha : float,
	AP : region(ispace(int2d), float),
	X : region(ispace(int1d), float),
	beta : float,
	Y : region(ispace(int1d), float))
where
	reads(AP),
	reads(X),
	reads writes(Y)
end

extern task sger(
	layout : int,
	alpha : float,
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float),
	A : region(ispace(int2d), float))
where
	reads(X),
	reads(Y),
	reads writes(A)
end

extern task ssyr(
	layout : int,
	Uplo : int,
	alpha : float,
	X : region(ispace(int1d), float),
	A : region(ispace(int2d), float))
where
	reads(X),
	reads writes(A)
end

extern task sspr(
	layout : int,
	Uplo : int,
	N : int,
	alpha : float,
	X : region(ispace(int1d), float),
	AP : region(ispace(int2d), float))
where
	reads(X),
	reads writes(AP)
end

extern task ssyr2(
	layout : int,
	Uplo : int,
	alpha : float,
	X : region(ispace(int1d), float),
	Y : region(ispace(int1d), float),
	A : region(ispace(int2d), float))
where
	reads(X),
	reads(Y),
	reads writes(A)
end

extern task sspr2(
	layout : int,
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
end

extern task dsymv(
	layout : int,
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
end

extern task dsbmv(
	layout : int,
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
end

extern task dspmv(
	layout : int,
	Uplo : int,
	N : int,
	alpha : double,
	AP : region(ispace(int2d), double),
	X : region(ispace(int1d), double),
	beta : double,
	Y : region(ispace(int1d), double))
where
	reads(AP),
	reads(X),
	reads writes(Y)
end

extern task dger(
	layout : int,
	alpha : double,
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double),
	A : region(ispace(int2d), double))
where
	reads(X),
	reads(Y),
	reads writes(A)
end

extern task dsyr(
	layout : int,
	Uplo : int,
	alpha : double,
	X : region(ispace(int1d), double),
	A : region(ispace(int2d), double))
where
	reads(X),
	reads writes(A)
end

extern task dspr(
	layout : int,
	Uplo : int,
	N : int,
	alpha : double,
	X : region(ispace(int1d), double),
	AP : region(ispace(int2d), double))
where
	reads(X),
	reads writes(AP)
end

extern task dsyr2(
	layout : int,
	Uplo : int,
	alpha : double,
	X : region(ispace(int1d), double),
	Y : region(ispace(int1d), double),
	A : region(ispace(int2d), double))
where
	reads(X),
	reads(Y),
	reads writes(A)
end

extern task dspr2(
	layout : int,
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
end

extern task sgemm(
	layout : int,
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
end

extern task ssymm(
	layout : int,
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
end

extern task ssyrk(
	layout : int,
	Uplo : int,
	Trans : int,
	alpha : float,
	A : region(ispace(int2d), float),
	beta : float,
	C : region(ispace(int2d), float))
where
	reads(A),
	reads writes(C)
end

extern task ssyr2k(
	layout : int,
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
end

extern task strmm(
	layout : int,
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
end

extern task strsm(
	layout : int,
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
end

extern task dgemm(
	layout : int,
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
end

extern task dsymm(
	layout : int,
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
end

extern task dsyrk(
	layout : int,
	Uplo : int,
	Trans : int,
	alpha : double,
	A : region(ispace(int2d), double),
	beta : double,
	C : region(ispace(int2d), double))
where
	reads(A),
	reads writes(C)
end

extern task dsyr2k(
	layout : int,
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
end

extern task dtrmm(
	layout : int,
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
end

extern task dtrsm(
	layout : int,
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
end
