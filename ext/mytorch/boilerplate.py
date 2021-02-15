from pathlib import Path


class Datatype(object):
    def __init__(self, name, pass_to_kernel=True):
        self.name = name
        self.pass_to_kernel = pass_to_kernel


class Primitive(Datatype):
    def __init__(self, name, dtype=None, **kwargs):
        super().__init__(name, **kwargs)
        self.dtype = dtype

    def get_function_argument(self):
        return f"{self.dtype} {self.name}"

    def get_dispatch_conv(self):
        return self.name


class Component(object):
    def __init__(self):
        pass

    def __str__(self):
        return "#######"

    def shape_str(self):
        return "#######"


class FixedComponent(Component):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def shape_str(self):
        return str(self.size)


class NamedComponent(Component):
    def __init__(self, name, multiplier=1):
        super().__init__()
        self.name = name
        self.multiplier = multiplier

    def __str__(self):
        return self.name

    def shape_str(self):
        if self.multiplier == 1:
            return self.name
        else:
            return f"{self.multiplier} * {self.name}"


class Tensor(Datatype):
    def __init__(
        self,
        name,
        components,
        dtype=None,
        dispatch_dtype="scalar_t",
        factory="zeros",
        factory_fill_value=-1,
        device=None,
        requires_grad=None,
        intermediate=False,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.parse_components(components)
        self.dtype = dtype
        self.dispatch_dtype = dispatch_dtype
        self.factory = factory
        self.factory_fill_value = factory_fill_value
        self.device = device
        self.requires_grad = requires_grad
        self.intermediate = intermediate

    def parse_components(self, components):
        self.components = []
        for comp in components:
            if isinstance(comp, Component):
                self.components.append(comp)
            elif isinstance(comp, int):
                self.components.append(FixedComponent(size=comp))
            else:
                self.components.append(NamedComponent(name=comp))

    def get_function_argument(self):
        return f"at::Tensor {self.name}"

    def get_dispatch_conv(self):
        return f"torch2co{len(self.components)}<{self.dispatch_dtype}>({self.name})"

    def get_dtype(self, input):
        if self.dtype is None:
            return f"{input[0].name}.dtype()"
        elif self.dtype in [x.name for x in input]:
            return f"{self.dtype}.dtype()"
        else:
            return self.dtype

    def get_factory(self, shape, options):
        if self.factory == "full":
            return f"torch::full({{{shape}}}, {self.factory_fill_value}, {options})"
        else:
            return f"torch::{self.factory}({{{shape}}}, {options})"


class Kernel(object):
    def __init__(
        self,
        class_name=None,
        params=None,
        length=None,
        cuda_num_threads=None,
        dispatch_scalar_type_name=None,
    ):
        self.class_name = class_name
        self.params = params
        self.length = length
        self.cuda_num_threads = cuda_num_threads
        self.dispatch_scalar_type_name = dispatch_scalar_type_name

    def write_kernel(self, fp, class_name, input, output, engine):
        if self.class_name is not None:
            class_name = self.class_name
        dispatch_scalar_type_name = self.dispatch_scalar_type_name
        if dispatch_scalar_type_name is None:
            dispatch_scalar_type_name = output[0].name
        fp.write(
            f'    AT_DISPATCH_FLOATING_TYPES({dispatch_scalar_type_name}.scalar_type(), "{class_name}", ([&] {{ \n'
        )
        fp.write(f"        iterate_{engine}({class_name}<scalar_t>(\n")
        if self.params is None:
            params = input + output
        else:
            params = []
            for param_name in self.params:
                if isinstance(param_name, Datatype):
                    params.append(param_name)
                else:
                    for param in input + output:
                        if param_name == param.name:
                            params.append(param)
                            break
        params = ",\n".join(
            [
                " " * 12 + param.get_dispatch_conv()
                for param in params
                if param.pass_to_kernel
            ]
        )
        fp.write(params)
        fp.write(f")")
        length_str = []
        for comp_name in self.length:
            comp_found = False
            for param in input:
                if isinstance(param, Tensor):
                    found = False
                    for comp_idx, param_comp in enumerate(param.components):
                        if comp_name == str(param_comp):
                            length_str.append(f"{param.name}.size({comp_idx})")
                            found = True
                            comp_found = True
                            break
                    if found:
                        break
                elif isinstance(param, Primitive):
                    if comp_name == param.name:
                        length_str.append(param.name)
                        comp_found = True
            if not comp_found:
                raise Exception(
                    f"{class_name} did not found {comp_name} for length_str"
                )
        length_str = " * ".join(length_str)
        fp.write(f", {length_str}")
        if engine == "cuda" and self.cuda_num_threads is not None:
            fp.write(f", {self.cuda_num_threads}")
        fp.write(");\n")
        fp.write("    }));\n\n")


class Function(object):
    def __init__(
        self,
        *,
        name,
        fwd_input,
        fwd_output,
        fwd_kernels,
        bwd_input=None,
        bwd_output=None,
        bwd_kernels=None,
    ):
        self.name = name
        self.fwd_input = fwd_input
        self.fwd_output = fwd_output
        self.fwd_kernels = fwd_kernels
        self.bwd_input = [] if bwd_input is None else bwd_input
        self.bwd_output = [] if bwd_output is None else bwd_output
        self.bwd_kernels = [] if bwd_kernels is None else bwd_kernels

        self.out_dir = Path("./generated/")
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.generate()

    def has_backward(self):
        return (
            len(self.bwd_input) > 0
            and len(self.bwd_output) > 0
            and len(self.bwd_kernels) > 0
        )

    def get_class_name(self, type=""):
        if type == "":
            name = f"{self.name}"
        else:
            name = f"{self.name}_{type}"
        return "".join(x.capitalize() or "_" for x in name.split("_"))

    def generate(self):
        self.generate_cpu()
        self.generate_cuda()
        self.generate_kernel()

    def get_return_type(self, output):
        if len(output) == 1:
            return "at::Tensor"
        else:
            return "std::vector<at::Tensor>"

    def generate_fun_definition(self, fp, type, input, output):
        arguments = ", ".join(
            [param.get_function_argument() for param in input]
        )
        fp.write(
            f"{self.get_return_type(output)} {self.name}_{type}({arguments})"
        )

    def generate_kernel_definition(self, fp, type, input, output):
        arguments = ", ".join(
            [param.get_function_argument() for param in input + output]
        )
        fp.write(f"void {self.name}_{type}_kernel({arguments})")

    def generate_param_checks(self, fp, engine, input, output):
        defined_components = {}
        for param_idx, param in enumerate(input):
            if not isinstance(param, Tensor):
                continue
            fp.write(
                f"    CHECK_INPUT_{engine}_DIM({param.name}, {len(param.components)});\n"
            )
            for component_idx, component in enumerate(param.components):
                if isinstance(component, FixedComponent):
                    fp.write(
                        f'    AT_ASSERTM({param.name}.size({component_idx}) == {component.size}, "{param.name} has wrong shape, should be {component.size} on dim={component_idx}");\n'
                    )
                elif component.name in defined_components:
                    fp.write(
                        f'    AT_ASSERTM({param.name}.size({component_idx}) == {component.multiplier} * {component.name}, "{component.name} of {param.name} does not match");\n'
                    )
                else:
                    other_usage = False
                    for other_param_idx in range(param_idx + 1, len(input)):
                        other_param = input[other_param_idx]
                        other_usage = other_usage or (
                            isinstance(other_param, Tensor)
                            and component.name
                            in [
                                comp.name
                                for comp in other_param.components
                                if isinstance(comp, NamedComponent)
                            ]
                        )
                    for other_param in output:
                        other_usage = other_usage or component.name in [
                            comp.name
                            for comp in other_param.components
                            if isinstance(comp, NamedComponent)
                        ]
                    if other_usage:
                        defined_components[component.name] = True
                        mul = (
                            ""
                            if component.multiplier == 1
                            else f" / {component.multiplier}"
                        )
                        fp.write(
                            f"    const long {component} = {param.name}.size({component_idx}){mul};\n"
                        )
        fp.write("\n")

    def generate_outputs(self, fp, input, output):
        for param in output:
            shape = ", ".join([comp.shape_str() for comp in param.components])
            dtype = param.get_dtype(input)
            if param.device is None:
                device = f"{input[0].name}.device().type()"
            elif param.device in [x.name for x in input]:
                device = f"{device}.device().type()"
            else:
                device = param.device
            if param.intermediate:
                requires_grad = "false"
            elif param.requires_grad is None:
                requires_grad = f"{input[0].name}.requires_grad()"
            elif param.requires_grad in [x.name for x in input]:
                requires_grad = f"{requires_grad}.requires_grad()"
            else:
                requires_grad = param.requires_grad
            options = f"torch::TensorOptions().dtype({dtype}).device({device}).requires_grad({requires_grad})"
            fp.write(
                f"    auto {param.name} = {param.get_factory(shape, options)};\n"
            )
        fp.write("\n")

    def generate_kernel_call(self, fp, type, input, output):
        args = ", ".join([param.name for param in input + output])
        fp.write(f"    {self.name}_{type}_kernel({args});\n\n")

    def generate_returns(self, fp, output):
        if len(output) == 1:
            fp.write(f"    return {output[0].name};\n")
        else:
            ret = ", ".join([param.name for param in output])
            fp.write(f"    return {{{ret}}};\n")

    def generate_pybind_init_fun(self, fp):
        fp.write(f"void {self.name}_pybind_init(py::module& m) {{\n")
        fp.write(f'    m.def("{self.name}_forward", &{self.name}_forward);\n')
        if self.has_backward():
            fp.write(
                f'    m.def("{self.name}_backward", &{self.name}_backward);\n'
            )
        fp.write("}\n")

    def generate_cpu(self):
        out_path = self.out_dir / f"{self.name}_cpu.cpp"
        print(f"generate {out_path}")
        with open(out_path, "w") as fp:
            fp.write(f"#include <common_cpu.h>\n")
            fp.write(f'#include "{self.name}.h"\n\n')

            self.generate_fun_definition(
                fp, type="forward", input=self.fwd_input, output=self.fwd_output
            )
            fp.write("{\n")
            self.generate_param_checks(
                fp, engine="CPU", input=self.fwd_input, output=self.fwd_output
            )
            self.generate_outputs(
                fp, input=self.fwd_input, output=self.fwd_output
            )
            for kernel in self.fwd_kernels:
                kernel.write_kernel(
                    fp,
                    self.get_class_name(type="forward"),
                    input=self.fwd_input,
                    output=self.fwd_output,
                    engine="cpu",
                )
            self.generate_returns(fp, output=self.fwd_output)
            fp.write("}\n\n")

            if self.has_backward():
                self.generate_fun_definition(
                    fp,
                    type="backward",
                    input=self.bwd_input,
                    output=self.bwd_output,
                )
                fp.write("{\n")
                self.generate_param_checks(
                    fp,
                    engine="CPU",
                    input=self.bwd_input,
                    output=self.bwd_output,
                )
                self.generate_outputs(
                    fp, input=self.bwd_input, output=self.bwd_output
                )
                for kernel in self.bwd_kernels:
                    kernel.write_kernel(
                        fp,
                        self.get_class_name(type="backward"),
                        input=self.bwd_input,
                        output=self.bwd_output,
                        engine="cpu",
                    )
                self.generate_returns(fp, output=self.bwd_output)
                fp.write("}\n\n")

            self.generate_pybind_init_fun(fp)

    def generate_cuda(self):
        out_path = self.out_dir / f"{self.name}_cuda.cpp"
        print(f"generate {out_path}")
        with open(out_path, "w") as fp:
            fp.write(f'#include "{self.name}.h"\n\n')

            self.generate_kernel_definition(
                fp, type="forward", input=self.fwd_input, output=self.fwd_output
            )
            fp.write(";\n\n")
            self.generate_fun_definition(
                fp, type="forward", input=self.fwd_input, output=self.fwd_output
            )
            fp.write("{\n")
            self.generate_param_checks(
                fp, engine="CUDA", input=self.fwd_input, output=self.fwd_output
            )
            self.generate_outputs(
                fp, input=self.fwd_input, output=self.fwd_output
            )
            self.generate_kernel_call(
                fp, type="forward", input=self.fwd_input, output=self.fwd_output
            )
            self.generate_returns(fp, output=self.fwd_output)
            fp.write("}\n\n")

            if self.has_backward():
                self.generate_kernel_definition(
                    fp,
                    type="backward",
                    input=self.bwd_input,
                    output=self.bwd_output,
                )
                fp.write(";\n\n")
                self.generate_fun_definition(
                    fp,
                    type="backward",
                    input=self.bwd_input,
                    output=self.bwd_output,
                )
                fp.write("{\n")
                self.generate_param_checks(
                    fp,
                    engine="CUDA",
                    input=self.bwd_input,
                    output=self.bwd_output,
                )
                self.generate_outputs(
                    fp, input=self.bwd_input, output=self.bwd_output
                )
                self.generate_kernel_call(
                    fp,
                    type="backward",
                    input=self.bwd_input,
                    output=self.bwd_output,
                )
                self.generate_returns(fp, output=self.bwd_output)
                fp.write("}\n\n")

            self.generate_pybind_init_fun(fp)

    def generate_kernel(self):
        out_path = self.out_dir / f"{self.name}_kernel.cu"
        print(f"generate {out_path}")
        with open(out_path, "w") as fp:
            fp.write(f"#include <common_cuda.h>\n")
            fp.write(f"#include <torch_common.h>\n")
            fp.write(f'#include "{self.name}.h"\n\n')

            self.generate_kernel_definition(
                fp, type="forward", input=self.fwd_input, output=self.fwd_output
            )
            fp.write("{\n")
            for kernel in self.fwd_kernels:
                kernel.write_kernel(
                    fp,
                    self.get_class_name(type="forward"),
                    input=self.fwd_input,
                    output=self.fwd_output,
                    engine="cuda",
                )
            fp.write("}\n")

            if self.has_backward():
                self.generate_kernel_definition(
                    fp,
                    type="backward",
                    input=self.bwd_input,
                    output=self.bwd_output,
                )
                fp.write("{\n")
                for kernel in self.bwd_kernels:
                    kernel.write_kernel(
                        fp,
                        self.get_class_name(type="backward"),
                        input=self.bwd_input,
                        output=self.bwd_output,
                        engine="cuda",
                    )
                fp.write("}\n")

    def generate_python(self, fp, ext):
        fwd_args = ", ".join([param.name for param in self.fwd_input])
        fwd_args_cont = []
        for param in self.fwd_input:
            if isinstance(param, Tensor):
                fwd_args_cont.append(f"{param.name}.contiguous()")
            else:
                fwd_args_cont.append(f"{param.name}")
        fwd_args_cont = ", ".join(fwd_args_cont)
        fwd_output = ", ".join([param.name for param in self.fwd_output])
        return_output = ", ".join(
            [param.name for param in self.fwd_output if not param.intermediate]
        )
        bwd_args_cont = []
        for param in self.bwd_input:
            if isinstance(param, Tensor):
                bwd_args_cont.append(f"{param.name}.contiguous()")
            else:
                bwd_args_cont.append(f"{param.name}")
        bwd_args_cont = ", ".join(bwd_args_cont)
        bwd_output = ", ".join([param.name for param in self.bwd_output])
        grad_output = ", ".join(
            [
                f"grad_{param.name}"
                for param in self.fwd_output
                if not param.intermediate
            ]
        )
        if len(fwd_output) > 0:
            fwd_output = fwd_output + " ="
        fp.write(
            f"""class {self.get_class_name()}(torch.autograd.Function):
  @staticmethod
  def forward(ctx, {fwd_args}):
    args = ({fwd_args_cont})
    if {self.fwd_input[0].name}.is_cuda:
      {fwd_output} {ext}_cuda.{self.name}_forward(*args)
    else:
      {fwd_output} {ext}_cpu.{self.name}_forward(*args)
"""
        )
        saved_tensors = []
        for param in self.fwd_input + self.fwd_output:
            if param.name in [p.name for p in self.bwd_input]:
                if isinstance(param, Tensor):
                    saved_tensors.append(param.name)
                else:
                    fp.write(f"    ctx.{param.name} = {param.name}\n")
        saved_tensors = ", ".join(saved_tensors)
        if len(saved_tensors) > 0:
            fp.write(f"    ctx.save_for_backward({saved_tensors})\n")
        if len(return_output) == 0:
            # TODO: find a real solution
            fp.write(f"    return out\n\n")
        else:
            fp.write(f"    return {return_output}\n\n")

        fp.write("  @staticmethod\n")
        fp.write(f"  def backward(ctx, {grad_output}):\n")
        for param in self.fwd_input:
            fp.write(f"    grad_{param.name} = None\n")
        if self.has_backward():
            if len(saved_tensors) > 0:
                fp.write(f"    {saved_tensors}, = ctx.saved_tensors\n")
            for param in self.fwd_input + self.fwd_output:
                if not isinstance(param, Tensor) and param.name in [
                    p.name for p in self.bwd_input
                ]:
                    fp.write(f"    {param.name} = ctx.{param.name}\n")
            fp.write(
                f"""    args = ({bwd_args_cont})
    if {self.bwd_input[0].name}.is_cuda:
      {bwd_output} = {ext}_cuda.{self.name}_backward(*args)
    else:
      {bwd_output} = {ext}_cpu.{self.name}_backward(*args)"""
            )
        ret = ", ".join([f"grad_{param.name}" for param in self.fwd_input])
        fp.write(f"\n    return {ret}")

        fp.write("\n\n")
        fp.write(f"def {self.name}({fwd_args}):\n")
        fp.write(f"  return {self.get_class_name()}.apply({fwd_args})")
        fp.write("\n\n\n")


class Project(object):
    def __init__(self, name, functions):
        self.name = name
        self.functions = functions

        self.out_dir = Path("./generated/")
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.generate_cpu()
        self.generate_cuda()
        self.generate_kernel()
        self.generate_python()

    def get_cpu_path(self):
        return self.out_dir / f"{self.name}_cpu.cpp"

    def get_cuda_path(self):
        return self.out_dir / f"{self.name}_cuda.cpp"

    def get_kernel_path(self):
        return self.out_dir / f"{self.name}_kernel.cu"

    def generate_cpu(self):
        out_path = self.get_cpu_path()
        print(f"generate {out_path}")
        with open(out_path, "w") as fp:
            fp.write(f"#include <torch_common.h>\n")
            for function in self.functions:
                fp.write(f'#include "{function.name}_cpu.cpp"\n')
            fp.write("\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n")
            for function in self.functions:
                fp.write(f"    {function.name}_pybind_init(m);\n")
            fp.write("}\n")

    def generate_cuda(self):
        out_path = self.get_cuda_path()
        print(f"generate {out_path}")
        with open(out_path, "w") as fp:
            fp.write(f"#include <torch_common.h>\n")
            for function in self.functions:
                fp.write(f'#include "{function.name}_cuda.cpp"\n')
            fp.write("\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n")
            for function in self.functions:
                fp.write(f"    {function.name}_pybind_init(m);\n")
            fp.write("}\n")

    def generate_kernel(self):
        out_path = self.get_kernel_path()
        print(f"generate {out_path}")
        with open(out_path, "w") as fp:
            for function in self.functions:
                fp.write(f'#include "{function.name}_kernel.cu"\n')

    def generate_python(self):
        out_path = f"generated_{self.name}.py"
        print(f"generate {out_path}")
        with open(out_path, "w") as fp:
            fp.write("import torch\n")
            fp.write(
                f"""
try:
    from . import {self.name}_cpu
except ImportError:
    try:
        import {self.name}_cpu
    except ImportError as err:
        print("[WARNING] could not load {self.name}_cpu in generated_ext.py")
        print(err)
try:
    from . import {self.name}_cuda
except ImportError:
    try:
        import {self.name}_cuda
    except ImportError as err:
        print("[WARNING] could not load {self.name}_cuda in generated_ext.py")
        print(err)\n\n"""
            )
            for function in self.functions:
                function.generate_python(fp, ext=self.name)
