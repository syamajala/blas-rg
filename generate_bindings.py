"""
Copyright 2018 Stanford University

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

from collections import namedtuple
from pygccxml import utils
from pygccxml import declarations
from pygccxml import parser
from template import header

Arg = namedtuple('Arg', ['name', 'type', 'pointer', 'const'])
blas_types = ['s', 'd']
#blas_types = ['s', 'd', 'c', 'z', 'cs', 'zd', 'sc', 'dz']
vectors = ["X", "Y"]
matrices = ["A", "B", "C"]

# Handle enums
enum_names = ["CBLAS_LAYOUT", "CBLAS_TRANSPOSE", "CBLAS_UPLO", "CBLAS_DIAG", "CBLAS_SIDE"]
# enums = {}
# for name in enum_names:
#     matcher = declarations.declaration_matcher(name=name, decl_type=declarations.enumeration.enumeration_t)
#     enum = declarations.matcher.get_single(matcher, global_namespace)
#     enums[name] = enum


def sanitize_enum_name(name):
    for enum_name in enum_names:
        if enum_name in name:
            return enum_name


class Func():

    def __init__(self, name):
        self.name = name
        self.args = []

    def __repr__(self):
        return self.name

    def to_terra(self):
        terra_args = []
        body = []

        pointer_args = {}
        prev_type_pointer = False
        for arg in self.args:
            if arg.name == "layout":
                continue

            if arg.pointer:
                arg_str = "rect%s : rect%dd"
                if arg.name in vectors:
                    d = 1
                elif arg.name in matrices:
                    d = 2

                terra_args.append(arg_str % (arg.name, d))
                pointer_args[arg.name] = d
                prev_type_pointer = True
            elif prev_type_pointer:
                prev_type_pointer = False
            else:
                terra_args.append("%s : %s" % (arg.name, arg.type))

        for arg in pointer_args:
            terra_args.append("pr%s : c.legion_physical_region_t" % arg)
            terra_args.append("fld%s : c.legion_field_id_t" % arg)

        for arg, dim  in pointer_args.items():
            body.append("var raw%s = get_raw_pointer_%dd(rect%s, pr%s, fld%s)" % (arg, dim, arg, arg, arg))

        cblas_args = []
        prev_type_pointer = False
        offset_name = ""
        for arg in self.args:
            if arg.name == "layout":
                cblas_args.append("cblas.CblasColMajor")
            elif arg.pointer:
                prev_type_pointer = True
                cblas_args.append("raw%s.ptr" % arg.name)
                offset_name = arg.name
            elif prev_type_pointer:
                prev_type_pointer = False
                cblas_args.append("raw%s.offset" % offset_name)
                offset_name = ""
            else:
                cblas_args.append(arg.name)

        cblas_call = "cblas.%s(%s)"
        cblas_args = ", ".join(cblas_args)

        terra_args = list(map(lambda a: "\n\t"+a, terra_args))
        terra_args = "".join(terra_args)
        body.append(cblas_call % (self.name, cblas_args))
        body = list(map(lambda b: "\t"+b, body))
        body = "\n".join(body)

        terra_func = "terra %s_terra(%s)\n\n%s\nend\n\n" % (self.name, terra_args, body)
        return terra_func

    def to_regent():
        pass


def parse_funcs(ns):
    funcs = []

    for func in ns.free_functions():
        name = func.name.split("_")[1]

        if not name[0] in blas_types:
            print("Unsupported blas type", name)
            continue

        f = Func(name)
        skip = False

        for a in func.arguments:
            if declarations.type_traits.is_fundamental(a.decl_type):
                typ = declarations.type_traits.remove_const(a.decl_type).decl_string
                const = declarations.type_traits.is_const(a.decl_type)
                f.args.append(Arg(a.name, typ, False, const))
            elif declarations.type_traits.is_pointer(a.decl_type):
                typ = declarations.type_traits.remove_pointer(a.decl_type)
                const = declarations.type_traits.is_const(typ)
                base_typ = declarations.type_traits.base_type(a.decl_type)
                if declarations.type_traits.is_void(base_typ):
                    skip = True
                    print("Skipping %s due to void * argument %s" % (func.name, a.name))
                    break
                f.args.append(Arg(a.name, base_typ.decl_string, True, const))
            elif declarations.type_traits_classes.is_enum(a.decl_type):
                decl_string = sanitize_enum_name(a.decl_type.decl_string)
                if decl_string in enum_names:
                    f.args.append(Arg(a.name, 'int', False, False))
                else:
                    raise TypeError('Function: %s unknown enum %s' % (func.name, a.name), a.decl_type)
            else:
                raise TypeError("Function: %s has an argument: %s of unhandled type" % (func.name, a.name), a.decl_type)

        if not skip:
            print("Parsed", func.name)
            funcs.append(f)

    return funcs


if __name__ == '__main__':
    # Find the location of the xml generator (castxml or gccxml)
    generator_path, generator_name = utils.find_xml_generator()

    # Configure the xml generator
    xml_generator_config = parser.xml_generator_configuration_t(
        xml_generator_path=generator_path,
        xml_generator=generator_name)

    # The c++ file we want to parse
    filename = "/usr/include/cblas.h"

    # Parse the c++ file
    decls = parser.parse([filename], xml_generator_config)

    # Get access to the global namespace
    global_namespace = declarations.get_global_namespace(decls)

    funcs = parse_funcs(global_namespace)
    terra_funcs = []
    for func in funcs:
        try:
            terra_funcs.append(func.to_terra())
        except UnboundLocalError:
            print("Skipping:", func.name)

    with open("blas.rg", 'w') as f:
        f.write(header)
        f.writelines(terra_funcs)
