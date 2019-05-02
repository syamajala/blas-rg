"""
Copyright 2019 Stanford University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pygccxml import utils
from pygccxml import declarations
from pygccxml import parser
from numpy import f2py
from template import copyright, blas_header, task_template, task_template_no_priv
import re

vectors = ["X", "Y"]
matrices = ["A", "B", "C"]
enum_names = ["CBLAS_ORDER", "CBLAS_TRANSPOSE", "CBLAS_UPLO", "CBLAS_DIAG", "CBLAS_SIDE"]
blas_types = {'i': 'real', 's': 'float', 'd': 'double', 'c': 'complex', 'z': 'complex'}


def sanitize_enum_name(name):
    for enum_name in enum_names:
        if enum_name in name:
            return enum_name


def name2dim(name):
    other = {"PARAM": 1, "AP": 1, "ALPHA": 1,
             "DOTU": 1, "CDOTC": 1, "DOTC": 1,
             "B1": 1, "P": 1}
    if name in vectors:
        return 1
    elif name in matrices:
        return 2
    elif name in other:
        return other[name]
    else:
        raise KeyError("Unknown Name:", name)


def typespec2type(typespec):
    if typespec == 'real':
        return 'float'
    elif typespec == 'double' or typespec == 'double precision':
        return 'double'
    elif typespec == 'complex' or typespec == 'double complex':
        return 'complex'
    elif typespec == 'int' or typespec == 'integer':
        return 'int'
    elif typespec == 'float':
        return 'float'
    else:
        raise TypeError("Unknown Type:", typespec)


def find_var(name, fortran_vars={}, c_args=[], blas_type=None):
    for var in fortran_vars:
        if var == name.lower():
            for arg in c_args:
                if arg.name == name:
                    if declarations.type_traits.is_pointer(arg.decl_type):
                        if name in vectors or name in matrices:
                            fortran_vars[var]['dimension'] = True
                        else:
                            fortran_vars[var]['pointer'] = True
                        break
            return fortran_vars[var]

    for arg in c_args:
        if arg.name == name:
            var = {}

            base_typ = declarations.type_traits.base_type(arg.decl_type)
            if declarations.type_traits.is_void(base_typ):
                var['typespec'] = blas_type
                var['void'] = True
            else:
                var['typespec'] = str(base_typ)

            if declarations.type_traits_classes.is_enum(arg.decl_type):
                var['typespec'] = 'integer'

            if declarations.type_traits.is_pointer(arg.decl_type):
                var['dimension'] = True

            const = declarations.type_traits.is_const(arg.decl_type)
            if const:
                var['intent'] = ['in']
            else:
                var['intent'] = ['out']

            assert len(var) != 0
            fortran_vars[name.lower()] = var
            return var


def sub(val):

    val = re.sub(r'trans_', 'trans', val)

    shape_re = r'shape\((\w+),\s*(\w+)\)'
    shapes = re.findall(shape_re, val)
    for v, dim in shapes:
        if dim == '0':
            size = f'size{v.upper()}.x'
        elif dim == '1':
            size = f'size{v.upper()}.y'
        val = re.sub(shape_re, size, val, count=1)

    lens = re.findall(r'len\((\w+)\)', val)
    for ln in lens:
        if ln.upper() in vectors:
            size = f'size{ln.upper()}'
        else:
            size = f'size{ln.upper()}.x'
        val = re.sub(r'len\(%s\)' % ln, size, val, count=1)

    val = re.sub(r'abs\((\w+)\)', r'\1', val)
    return val


class Func():

    def __init__(self, name, fortran_sig, c_func):
        self.name = name
        self.blas_type = blas_types[name[0]]
        self.terra_args = []
        self.fortran_sig = fortran_sig
        self.c_func = c_func
        self.return_type = c_func.return_type.decl_string
        self.pointer_args = []

    def __repr__(self):
        return self.name

    def to_terra(self):
        cblas_args = []
        body = []
        pointer_args = self.pointer_args

        for idx, name in enumerate(self.fortran_sig['c_order']):

            if name == "Order" or name == "order":
                cblas_args.append("cblas.CblasColMajor")
                continue
            elif name.startswith('inc') or name.startswith('off') or name.startswith('ld'):
                assert 0 < idx
                prev_name = self.fortran_sig['c_order'][idx-1]
                prev_name = prev_name.upper()
                cblas_args.append(f"raw{prev_name}.offset")
                continue

            arg = find_var(name, self.fortran_sig['vars'], self.c_func.arguments, self.blas_type)

            assert arg is not None, "%s Unable to find variable: %s" % (self.name, name)

            if 'dimension' in arg:
                name = name.upper()
                arg_str = "rect%s : rect%dd"
                dim = name2dim(name)
                self.terra_args.append(arg_str % (name.upper(), dim))
                pointer_args.append(name)

                typ = typespec2type(arg['typespec'])
                body.append(f"var raw{name} : {typ}_ptr")
                body.append(f"[get_raw_ptr_factory({dim}, {typ}, rect{name}, pr{name}, fld{name}, raw{name}, {typ}_ptr)]")

                cblas_args.append(f"raw{name}.ptr")
            elif 'pointer' in arg:
                typ = typespec2type(arg['typespec'])
                self.terra_args.append(f"{name} : {typ}")
                cblas_args.append(f"&{name}")
            else:
                self.terra_args.append("%s : %s" % (name, typespec2type(arg['typespec'])))
                cblas_args.append(name)

        for arg in pointer_args:
            self.terra_args.append("pr%s : c.legion_physical_region_t" % arg)
            self.terra_args.append("fld%s : c.legion_field_id_t" % arg)

        cblas_call = ""

        if self.return_type != "void":
            cblas_call = "return "

        cblas_call += "cblas.cblas_%s(%s)"
        cblas_args = ", ".join(cblas_args)

        terra_args = list(map(lambda a: "\n\t"+a, self.terra_args))
        terra_args = ",".join(terra_args)
        body.append(cblas_call % (self.name, cblas_args))
        body = list(map(lambda b: "\t"+b, body))
        body = "\n".join(body)

        terra_func = "terra %s_terra(%s)\n\n%s\nend\n\n" % (self.name, terra_args, body)
        return terra_func

    def to_regent(self):
        task_args = []
        privileges = []
        body = []
        terra_args = []

        for name in self.pointer_args:

            var = find_var(name, self.fortran_sig['vars'])

            if 'intent' in var and 'out' in var['intent']:
                privileges.append(f"reads writes({name})")
            else:
                privileges.append(f"reads({name})")

            body.append(f"var rect{name} = {name}.bounds")
            dim = name2dim(name)
            if dim == 2:
                body.append(f"var size{name} = rect{name}.hi - rect{name}.lo + {{1, 1}}")
            elif dim == 1:
                body.append(f"var size{name} = rect{name}.hi - rect{name}.lo + {{1}}")
            else:
                raise KeyError("Unknown dimension for name:", name)

        for arg in self.terra_args:
            arg, typ = arg.split(':')
            name = arg.strip()
            typ = typ.strip()

            var = find_var(name, self.fortran_sig['vars'], self.c_func.arguments, self.blas_type)

            if var is None:
                if typ == 'c.legion_physical_region_t':
                    n = re.search(r'pr(\w+)', name)
                    n = n.groups()[0]
                    terra_args.append(f"__physical({n})[0]")
                elif typ == 'c.legion_field_id_t':
                    n = re.search(r'fld(\w+)', name)
                    n = n.groups()[0]
                    terra_args.append(f"__fields({n})[0]")
                else:
                    n = re.search(r'rect(\w+)', name)
                    n = n.groups()[0]
                    var = find_var(n, self.fortran_sig['vars'], self.c_func.arguments, self.blas_type)
                    dim = name2dim(n)
                    typ = typespec2type(var['typespec'])
                    task_args.append(f"{n} : region(ispace(int{dim}d), {typ})")
                    terra_args.append(name)
            elif var and 'depend' in var and '=' in var:
                val = var['=']

                val = sub(val)

                deps = set(var['depend'])
                deps = set(map(lambda v: re.sub(r'trans_', 'trans', v), deps))
                args = set(map(lambda a: a.lower(), self.fortran_sig['c_order']))
                args.discard('lda')
                args.discard('ldb')
                args.discard('ldc')
                args.discard('n')
                deps = deps.difference(args)

                dep_values = {'incx': '1', 'incy': '1'}
                for dep_name in deps:
                    dep = self.fortran_sig['vars'][dep_name]
                    if '=' in dep:
                        dep = sub(dep['='])
                        dep_values[dep_name] = dep

                for k, v in dep_values.items():
                    val = re.sub(k, v, val)

                trans = re.findall(r'(trans)(\w*)', val)
                for t, m in trans:
                    t = ''.join((t, m))
                    val = re.sub(t, f'Trans{m.upper()}', val)

                val = re.sub(r'side', 'Side', val)

                if '?' in val:
                    val = re.sub(r'\(', '', val, count=1)
                    val = re.sub(r'\)', '', val[::-1], count=1)
                    val = val[::-1]
                    cond, if_true, else_true = re.split(r'\?|\:', val)
                    cond = cond.strip()
                    if_true = if_true.strip()
                    else_true = else_true.strip()
                    body.append(f"var {name} = 0")
                    body.append(f"if [bool]({cond}) then\n\t\t{name} = {if_true}\n\telse\n\t\t{name} = {else_true}\n\tend\n")
                else:
                    body.append(f"var {name} = {val}")

                terra_args.append(name)
            else:
                task_args.append(f"{name} : {typ}")
                terra_args.append(name)

        task_args = list(map(lambda a: "\n\t"+a, task_args))
        task_args = ",".join(task_args)

        privileges = list(map(lambda p: "\t"+p, privileges))
        privileges = ',\n'.join(privileges)

        terra_call = self.name + "_terra(%s)"
        terra_args = ', '.join(terra_args)

        terra_call = terra_call % terra_args
        if self.return_type != 'void':
            terra_call = 'return ' + terra_call
        body.append(terra_call)

        body = list(map(lambda b: "\t"+b, body))
        body = "\n".join(body)

        if privileges:
            task = task_template % (self.name, task_args, privileges, body)
        else:
            task = task_template_no_priv % (self.name, task_args, body)
        return task


def parse_funcs(c_header, sig_file):
    # Find the location of the xml generator (castxml or gccxml)
    generator_path, generator_name = utils.find_xml_generator()

    # Configure the xml generator
    xml_generator_config = parser.xml_generator_configuration_t(
        xml_generator_path=generator_path,
        xml_generator=generator_name)

    # Parse the c++ file
    decls = parser.parse([c_header], xml_generator_config)

    # Get access to the global namespace
    ns = declarations.get_global_namespace(decls)

    sigs = f2py.crackfortran.crackfortran(sig_file)
    interface = sigs[0]['body'][0]

    def find_sig(name):
        for i in interface['body']:
            if i['name'] == name:
                if 'parent_block' in i:
                    del i['parent_block']
                return i

    funcs = []

    for func in ns.free_functions():
        name = func.name.split("_")[1]
        sig = find_sig(name)

        if sig is None:
            print("Unsupported blas funciton", name)
            continue

        f = Func(name, sig, func)
        sig['c_order'] = [arg.name for arg in func.arguments]
        funcs.append(f)

    return funcs


def find_func(func):
    for i in funcs:
        if i.name == func:
            return i

def generate_cpu_bindings():
    c_header = "cblas.h"
    sig_file = "fblas.pyf"

    funcs = parse_funcs(c_header, sig_file)
    terra_funcs = []
    regent_funcs = []
    bindings = []
    for idx, func in enumerate(funcs):
        terra = func.to_terra()
        if 'complex' in terra:
            continue

        terra_funcs.append(terra)
        regent = func.to_regent()
        regent_funcs.append(regent)
        bindings.append(func.name)

    with open("cblas.rg", 'w') as f:
        f.write(copyright)
        f.write(blas_header)
        f.writelines(terra_funcs)
        f.writelines(regent_funcs)

    return bindings
