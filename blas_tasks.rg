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

local cblas_h = utils.cblas
require("cblas")
if utils.use_gpu then
   require("cublas")
end

__demand(__cuda, __leaf)
task sdot(
    X : region(ispace(int1d), float),
    Y : region(ispace(int1d), float))
where
    reads(X),
    reads(Y)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       return sdot_gpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        return sdot_cpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task ddot(
    X : region(ispace(int1d), double),
    Y : region(ispace(int1d), double))
where
    reads(X),
    reads(Y)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       return ddot_gpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        return ddot_cpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task snrm2(
    X : region(ispace(int1d), float))
where
    reads(X)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       return snrm2_gpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        return snrm2_cpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task sasum(
    X : region(ispace(int1d), float))
where
    reads(X)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       return sasum_gpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        return sasum_cpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task dnrm2(
    X : region(ispace(int1d), double))
where
    reads(X)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       return dnrm2_gpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        return dnrm2_cpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task dasum(
    X : region(ispace(int1d), double))
where
    reads(X)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       return dasum_gpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        return dasum_cpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task isamax(
    X : region(ispace(int1d), float))
where
    reads(X)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       return isamax_gpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        return isamax_cpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task idamax(
    X : region(ispace(int1d), double))
where
    reads(X)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       return idamax_gpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        return idamax_cpu_terra(N, rectX, __physical(X)[0], __fields(X)[0])
    end
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
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       sswap_gpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        sswap_cpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
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
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       scopy_gpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        scopy_cpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
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
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       saxpy_gpu_terra(N, alpha, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        saxpy_cpu_terra(N, alpha, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
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
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       dswap_gpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dswap_cpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
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
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       dcopy_gpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dcopy_cpu_terra(N, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
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
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       daxpy_gpu_terra(N, alpha, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        daxpy_cpu_terra(N, alpha, rectX, rectY, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task srotg(
    a : float,
    b : float,
    c : float,
    s : float)
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       srotg_gpu_terra(a, b, c, s)
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        srotg_cpu_terra(a, b, c, s)
    end
end

__demand(__cuda, __leaf)
task srotmg(
    d1 : float,
    d2 : float,
    b1 : float,
    b2 : float,
    P : region(ispace(int1d), float))
where
    reads writes(P)
do
    var rectP = P.bounds
    var sizeP = rectP.hi - rectP.lo + {1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       srotmg_gpu_terra(d1, d2, b1, b2, rectP, __physical(P)[0], __fields(P)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        srotmg_cpu_terra(d1, d2, b1, b2, rectP, __physical(P)[0], __fields(P)[0])
    end
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
    var N = (sizeX-1-0)/1+1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       srot_gpu_terra(N, rectX, rectY, c, s, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        srot_cpu_terra(N, rectX, rectY, c, s, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task srotm(
    X : region(ispace(int1d), float),
    Y : region(ispace(int1d), float),
    P : region(ispace(int1d), float))
where
    reads writes(X),
    reads writes(Y),
    reads writes(P)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var rectP = P.bounds
    var sizeP = rectP.hi - rectP.lo + {1}
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       srotm_gpu_terra(N, rectX, rectY, rectP, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(P)[0], __fields(P)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        srotm_cpu_terra(N, rectX, rectY, rectP, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(P)[0], __fields(P)[0])
    end
end

__demand(__cuda, __leaf)
task drotg(
    a : double,
    b : double,
    c : double,
    s : double)
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       drotg_gpu_terra(a, b, c, s)
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        drotg_cpu_terra(a, b, c, s)
    end
end

__demand(__cuda, __leaf)
task drotmg(
    d1 : double,
    d2 : double,
    b1 : double,
    b2 : double,
    P : region(ispace(int1d), double))
where
    reads writes(P)
do
    var rectP = P.bounds
    var sizeP = rectP.hi - rectP.lo + {1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       drotmg_gpu_terra(d1, d2, b1, b2, rectP, __physical(P)[0], __fields(P)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        drotmg_cpu_terra(d1, d2, b1, b2, rectP, __physical(P)[0], __fields(P)[0])
    end
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
    var N = (sizeX-1-0)/1+1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       drot_gpu_terra(N, rectX, rectY, c, s, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        drot_cpu_terra(N, rectX, rectY, c, s, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task drotm(
    X : region(ispace(int1d), double),
    Y : region(ispace(int1d), double),
    P : region(ispace(int1d), double))
where
    reads writes(X),
    reads writes(Y),
    reads writes(P)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var rectP = P.bounds
    var sizeP = rectP.hi - rectP.lo + {1}
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       drotm_gpu_terra(N, rectX, rectY, rectP, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(P)[0], __fields(P)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        drotm_cpu_terra(N, rectX, rectY, rectP, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(P)[0], __fields(P)[0])
    end
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
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       sscal_gpu_terra(N, alpha, rectX, __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        sscal_cpu_terra(N, alpha, rectX, __physical(X)[0], __fields(X)[0])
    end
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
    var N = (sizeX-0)/1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote

                       dscal_gpu_terra(N, alpha, rectX, __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dscal_cpu_terra(N, alpha, rectX, __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task sgemv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var M = sizeA.x
    var N = sizeA.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       sgemv_gpu_terra(TransA - cblas_h.CblasNoTrans, M, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        sgemv_cpu_terra(layout, TransA, M, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task sgbmv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       sgbmv_gpu_terra(TransA - cblas_h.CblasNoTrans, M, N, KL, KU, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        sgbmv_cpu_terra(layout, TransA, M, N, KL, KU, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task strmv(
    layout : int,
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
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = sizeA.x
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       strmv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        strmv_cpu_terra(layout, Uplo, TransA, Diag, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task stbmv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = sizeA.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       stbmv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        stbmv_cpu_terra(layout, Uplo, TransA, Diag, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task stpmv(
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
do
    var rectAP = AP.bounds
    var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       stpmv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        stpmv_cpu_terra(layout, Uplo, TransA, Diag, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task strsv(
    layout : int,
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
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = sizeA.x
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       strsv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        strsv_cpu_terra(layout, Uplo, TransA, Diag, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task stbsv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = sizeA.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       stbsv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        stbsv_cpu_terra(layout, Uplo, TransA, Diag, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task stpsv(
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
do
    var rectAP = AP.bounds
    var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       stpsv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        stpsv_cpu_terra(layout, Uplo, TransA, Diag, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task dgemv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var M = sizeA.x
    var N = sizeA.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dgemv_gpu_terra(TransA - cblas_h.CblasNoTrans, M, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dgemv_cpu_terra(layout, TransA, M, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task dgbmv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dgbmv_gpu_terra(TransA - cblas_h.CblasNoTrans, M, N, KL, KU, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dgbmv_cpu_terra(layout, TransA, M, N, KL, KU, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task dtrmv(
    layout : int,
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
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = sizeA.x
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dtrmv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dtrmv_cpu_terra(layout, Uplo, TransA, Diag, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task dtbmv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = sizeA.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dtbmv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dtbmv_cpu_terra(layout, Uplo, TransA, Diag, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task dtpmv(
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
do
    var rectAP = AP.bounds
    var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dtpmv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dtpmv_cpu_terra(layout, Uplo, TransA, Diag, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task dtrsv(
    layout : int,
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
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = sizeA.x
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dtrsv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dtrsv_cpu_terra(layout, Uplo, TransA, Diag, N, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task dtbsv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var N = sizeA.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dtbsv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dtbsv_cpu_terra(layout, Uplo, TransA, Diag, N, K, rectA, rectX, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task dtpsv(
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
do
    var rectAP = AP.bounds
    var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dtpsv_gpu_terra(Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dtpsv_cpu_terra(layout, Uplo, TransA, Diag, N, rectAP, rectX, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0])
    end
end

__demand(__cuda, __leaf)
task ssymv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var N = sizeA.x
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       ssymv_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        ssymv_cpu_terra(layout, Uplo, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task ssbmv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var N = sizeA.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       ssbmv_gpu_terra(Uplo - cblas_h.CblasUpper, N, K, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        ssbmv_cpu_terra(layout, Uplo, N, K, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task sspmv(
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
do
    var rectAP = AP.bounds
    var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       sspmv_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectAP, rectX, beta, rectY, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        sspmv_cpu_terra(layout, Uplo, N, alpha, rectAP, rectX, beta, rectY, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task sger(
    layout : int,
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
    var M = sizeX
    var N = sizeY
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       sger_gpu_terra(M, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        sger_cpu_terra(layout, M, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
    end
end

__demand(__cuda, __leaf)
task ssyr(
    layout : int,
    Uplo : int,
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
    var N = (sizeX-1-0)/1+1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       ssyr_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectX, rectA, __physical(X)[0], __fields(X)[0], __physical(A)[0], __fields(A)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        ssyr_cpu_terra(layout, Uplo, N, alpha, rectX, rectA, __physical(X)[0], __fields(X)[0], __physical(A)[0], __fields(A)[0])
    end
end

__demand(__cuda, __leaf)
task sspr(
    layout : int,
    Uplo : int,
    N : int,
    alpha : float,
    X : region(ispace(int1d), float),
    AP : region(ispace(int2d), float))
where
    reads(X),
    reads writes(AP)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectAP = AP.bounds
    var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       sspr_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectX, rectAP, __physical(X)[0], __fields(X)[0], __physical(AP)[0], __fields(AP)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        sspr_cpu_terra(layout, Uplo, N, alpha, rectX, rectAP, __physical(X)[0], __fields(X)[0], __physical(AP)[0], __fields(AP)[0])
    end
end

__demand(__cuda, __leaf)
task ssyr2(
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
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var N = 0
    if [bool]((sizeX-1-0)/1+1 <=(sizeY-1-0)/1+1) then
        N = (sizeX-1-0)/1+1
    else
        N = (sizeY-1-0)/1+1
    end

    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       ssyr2_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        ssyr2_cpu_terra(layout, Uplo, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
    end
end

__demand(__cuda, __leaf)
task sspr2(
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
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       sspr2_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        sspr2_cpu_terra(layout, Uplo, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
    end
end

__demand(__cuda, __leaf)
task dsymv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var N = sizeA.x
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dsymv_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dsymv_cpu_terra(layout, Uplo, N, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task dsbmv(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var N = sizeA.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dsbmv_gpu_terra(Uplo - cblas_h.CblasUpper, N, K, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dsbmv_cpu_terra(layout, Uplo, N, K, alpha, rectA, rectX, beta, rectY, __physical(A)[0], __fields(A)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task dspmv(
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
do
    var rectAP = AP.bounds
    var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dspmv_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectAP, rectX, beta, rectY, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dspmv_cpu_terra(layout, Uplo, N, alpha, rectAP, rectX, beta, rectY, __physical(AP)[0], __fields(AP)[0], __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0])
    end
end

__demand(__cuda, __leaf)
task dger(
    layout : int,
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
    var M = sizeX
    var N = sizeY
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dger_gpu_terra(M, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dger_cpu_terra(layout, M, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
    end
end

__demand(__cuda, __leaf)
task dsyr(
    layout : int,
    Uplo : int,
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
    var N = (sizeX-1-0)/1+1
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dsyr_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectX, rectA, __physical(X)[0], __fields(X)[0], __physical(A)[0], __fields(A)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dsyr_cpu_terra(layout, Uplo, N, alpha, rectX, rectA, __physical(X)[0], __fields(X)[0], __physical(A)[0], __fields(A)[0])
    end
end

__demand(__cuda, __leaf)
task dspr(
    layout : int,
    Uplo : int,
    N : int,
    alpha : double,
    X : region(ispace(int1d), double),
    AP : region(ispace(int2d), double))
where
    reads(X),
    reads writes(AP)
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectAP = AP.bounds
    var sizeAP = rectAP.hi - rectAP.lo + {1, 1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dspr_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectX, rectAP, __physical(X)[0], __fields(X)[0], __physical(AP)[0], __fields(AP)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dspr_cpu_terra(layout, Uplo, N, alpha, rectX, rectAP, __physical(X)[0], __fields(X)[0], __physical(AP)[0], __fields(AP)[0])
    end
end

__demand(__cuda, __leaf)
task dsyr2(
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
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var N = 0
    if [bool]((sizeX-1-0)/1+1 <=(sizeY-1-0)/1+1) then
        N = (sizeX-1-0)/1+1
    else
        N = (sizeY-1-0)/1+1
    end

    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dsyr2_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dsyr2_cpu_terra(layout, Uplo, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
    end
end

__demand(__cuda, __leaf)
task dspr2(
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
do
    var rectX = X.bounds
    var sizeX = rectX.hi - rectX.lo + {1}
    var rectY = Y.bounds
    var sizeY = rectY.hi - rectY.lo + {1}
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dspr2_gpu_terra(Uplo - cblas_h.CblasUpper, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dspr2_cpu_terra(layout, Uplo, N, alpha, rectX, rectY, rectA, __physical(X)[0], __fields(X)[0], __physical(Y)[0], __fields(Y)[0], __physical(A)[0], __fields(A)[0])
    end
end

__demand(__cuda, __leaf)
task sgemm(
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

    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       sgemm_gpu_terra(TransA - cblas_h.CblasNoTrans, TransB - cblas_h.CblasNoTrans, M, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        sgemm_cpu_terra(layout, TransA, TransB, M, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
    end
end

__demand(__cuda, __leaf)
task ssymm(
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

    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       ssymm_gpu_terra(Side - cblas_h.CblasLeft, Uplo - cblas_h.CblasUpper, M, N, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        ssymm_cpu_terra(layout, Side, Uplo, M, N, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
    end
end

__demand(__cuda, __leaf)
task ssyrk(
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

    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       ssyrk_gpu_terra(Uplo - cblas_h.CblasUpper, Trans - cblas_h.CblasNoTrans, N, K, alpha, rectA, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(C)[0], __fields(C)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        ssyrk_cpu_terra(layout, Uplo, Trans, N, K, alpha, rectA, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(C)[0], __fields(C)[0])
    end
end

__demand(__cuda, __leaf)
task ssyr2k(
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

    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       ssyr2k_gpu_terra(Uplo - cblas_h.CblasUpper, Trans - cblas_h.CblasNoTrans, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        ssyr2k_cpu_terra(layout, Uplo, Trans, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
    end
end

__demand(__cuda, __leaf)
task strmm(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectB = B.bounds
    var sizeB = rectB.hi - rectB.lo + {1, 1}
    var M = sizeB.x
    var N = sizeB.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       strmm_gpu_terra(Side - cblas_h.CblasLeft, Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, M, N, alpha, rectA, rectB, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(B)[0], __fields(B)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        strmm_cpu_terra(layout, Side, Uplo, TransA, Diag, M, N, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
    end
end

__demand(__cuda, __leaf)
task strsm(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectB = B.bounds
    var sizeB = rectB.hi - rectB.lo + {1, 1}
    var M = sizeB.x
    var N = sizeB.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       strsm_gpu_terra(Side - cblas_h.CblasLeft, Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, M, N, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        strsm_cpu_terra(layout, Side, Uplo, TransA, Diag, M, N, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
    end
end

__demand(__cuda, __leaf)
task dgemm(
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

    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dgemm_gpu_terra(TransA - cblas_h.CblasNoTrans, TransB - cblas_h.CblasNoTrans, M, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dgemm_cpu_terra(layout, TransA, TransB, M, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
    end
end

__demand(__cuda, __leaf)
task dsymm(
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

    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dsymm_gpu_terra(Side - cblas_h.CblasLeft, Uplo - cblas_h.CblasUpper, M, N, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dsymm_cpu_terra(layout, Side, Uplo, M, N, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
    end
end

__demand(__cuda, __leaf)
task dsyrk(
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

    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dsyrk_gpu_terra(Uplo - cblas_h.CblasUpper, Trans - cblas_h.CblasNoTrans, N, K, alpha, rectA, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(C)[0], __fields(C)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dsyrk_cpu_terra(layout, Uplo, Trans, N, K, alpha, rectA, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(C)[0], __fields(C)[0])
    end
end

__demand(__cuda, __leaf)
task dsyr2k(
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

    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dsyr2k_gpu_terra(Uplo - cblas_h.CblasUpper, Trans - cblas_h.CblasNoTrans, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dsyr2k_cpu_terra(layout, Uplo, Trans, N, K, alpha, rectA, rectB, beta, rectC, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(C)[0], __fields(C)[0])
    end
end

__demand(__cuda, __leaf)
task dtrmm(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectB = B.bounds
    var sizeB = rectB.hi - rectB.lo + {1, 1}
    var M = sizeB.x
    var N = sizeB.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dtrmm_gpu_terra(Side - cblas_h.CblasLeft, Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, M, N, alpha, rectA, rectB, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0], __physical(B)[0], __fields(B)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dtrmm_cpu_terra(layout, Side, Uplo, TransA, Diag, M, N, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
    end
end

__demand(__cuda, __leaf)
task dtrsm(
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
do
    var rectA = A.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    var rectB = B.bounds
    var sizeB = rectB.hi - rectB.lo + {1, 1}
    var M = sizeB.x
    var N = sizeB.y
    var proc = get_executing_processor(__runtime())

    if clib.legion_processor_kind(proc) == clib.TOC_PROC then
        [(function()
            if utils.use_gpu then
                return rquote
                       regentlib.assert(layout == cblas_h.CblasColMajor, 'Expected column major layout.')
                       dtrsm_gpu_terra(Side - cblas_h.CblasLeft, Uplo - cblas_h.CblasUpper, TransA - cblas_h.CblasNoTrans, Diag - cblas_h.CblasNonUnit, M, N, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
                       end
            else
                return rquote regentlib.assert(false, "Build with CUDA support.") end
            end
        end)()]
    else
        dtrsm_cpu_terra(layout, Side, Uplo, TransA, Diag, M, N, alpha, rectA, rectB, __physical(A)[0], __fields(A)[0], __physical(B)[0], __fields(B)[0])
    end
end

local tasks_h = utils.output_dir .. "/blas_tasks.h"
local tasks_so = utils.output_dir .. "/blas_tasks.so"
regentlib.save_tasks(tasks_h, tasks_so, nil, nil, nil, nil, false)
