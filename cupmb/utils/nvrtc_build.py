#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, sys, functools, argparse

try:
    # cuda-python >= 12.6 (cuda-bindings)
    from cuda.bindings import driver as cuda, runtime as cudart, nvrtc
except ImportError:
    # legacy cuda-python layout
    from cuda import cuda, cudart, nvrtc

eprint = functools.partial(print, file=sys.stderr)


def cuda_get_error_enum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def check_cuda_error(result):
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={} ({})".format(result[0].value, cuda_get_error_enum(result[0]).decode("utf-8"))
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


CCE = check_cuda_error


def _extract_cuda_version(v):
    return f"{v // 1000}.{(v % 1000) // 10}.{v%10}"


def _encode_arg(a):
    return a.encode("ascii")


def _decode_arg(a):
    return a.decode("ascii")


def _read_lines(filename):
    with open(filename) as f:
        return tuple(filter(len, f.read().split(os.linesep)))


def _determine_target_architecture(major, minor):
    """
    Determine the target architecture and whether we can generate a cubin.

    A cubin is generated when NVRTC >= 11.1 and the device architecture is supported by NVRTC.
    Otherwise PTX is generated for the highest supported virtual architecture not greater than the
    device one, so that the driver can JIT compile it.
    """
    nvrtc_major, nvrtc_minor = CCE(nvrtc.nvrtcVersion())
    use_cubin = (nvrtc_major, nvrtc_minor) >= (11, 1)

    cc = 10 * major + minor
    try:
        supported_archs = tuple(map(int, CCE(nvrtc.nvrtcGetSupportedArchs())))
    except (AttributeError, RuntimeError):
        # nvrtcGetSupportedArchs is only available since NVRTC 11.2
        supported_archs = ()

    if supported_archs and (cc not in supported_archs):
        candidates = tuple(a for a in supported_archs if a <= cc)
        if not candidates:
            msg = (
                f"Device compute capability {major}.{minor} is not supported by NVRTC "
                f"{nvrtc_major}.{nvrtc_minor} (supported architectures are {supported_archs})."
            )
            raise RuntimeError(msg)
        eprint(
            f"Warning: device compute capability {major}.{minor} is not supported by NVRTC "
            f"{nvrtc_major}.{nvrtc_minor}, falling back to PTX for compute_{max(candidates)}."
        )
        cc = max(candidates)
        use_cubin = False

    return cc, use_cubin


def main(args):

    # Check Python-Cuda driver and cuda version
    int_driver_version = CCE(cudart.cudaDriverGetVersion())
    int_cuda_version = CCE(cudart.cudaRuntimeGetVersion())
    driver_version = _extract_cuda_version(int_driver_version)
    cuda_version = _extract_cuda_version(int_cuda_version)

    if int_driver_version < 11000:
        msg = (
            "You are using an outdated NVIDIA driver, expected a CUDA >= 11.0 compatible driver but found version "
            f"{driver_version} ({int_driver_version}). Please upgrade your NVidia driver to the latest stable "
            "driver supported by your device at https://www.nvidia.fr/Download/index.aspx."
        )
        eprint(msg)
        return 1

    if int_cuda_version < 11000:
        msg = (
            "You are using an outdated CUDA version, expected at least CUDA 11.0 but found version "
            f"{cuda_version} ({int_cuda_version}). Please download and install lastest CUDA release from "
            "https://developer.nvidia.com/cuda-downloads and rebuild Python-Cuda."
        )
        eprint(msg)
        return 1

    major = args.major
    minor = args.minor

    try:
        cc, use_cubin = _determine_target_architecture(major, minor)
    except RuntimeError as e:
        eprint(e)
        return 1

    prefix = "sm" if use_cubin else "compute"
    arch_arg = f"--gpu-architecture={prefix}_{cc}"

    build_options = args.build_options
    build_options += (arch_arg,)

    with open(args.source_file) as f:
        code = f.read()

    prog = CCE(nvrtc.nvrtcCreateProgram(_encode_arg(code), _encode_arg(args.source_file), 0, [], []))
    for kname in map(_encode_arg, args.kernel_names):
        CCE(nvrtc.nvrtcAddNameExpression(prog, kname))

    try:
        CCE(nvrtc.nvrtcCompileProgram(prog, len(build_options), list(map(_encode_arg, build_options))))
    except RuntimeError as e:
        log_size = CCE(nvrtc.nvrtcGetProgramLogSize(prog))
        log = b" " * log_size
        CCE(nvrtc.nvrtcGetProgramLog(prog, log))
        log = log.rstrip(b"\x00 ")
        eprint(f"Failed to compile source file {args.source_file}, build options were {build_options}.")
        eprint(_decode_arg(log))
        eprint(e)
        return 1

    mangled_kernel_names = []
    for kname in map(_encode_arg, args.kernel_names):
        mangled_name = CCE(nvrtc.nvrtcGetLoweredName(prog, kname))
        mangled_kernel_names.append(_decode_arg(mangled_name))

    filename = args.output_knames
    with open(filename, "w") as f:
        for mname in mangled_kernel_names:
            f.write(mname)
            f.write(os.linesep)

    if use_cubin:
        dataSize = CCE(nvrtc.nvrtcGetCUBINSize(prog))
        data = b" " * dataSize
        CCE(nvrtc.nvrtcGetCUBIN(prog, data))
    else:
        dataSize = CCE(nvrtc.nvrtcGetPTXSize(prog))
        data = b" " * dataSize
        CCE(nvrtc.nvrtcGetPTX(prog, data))

    filename = args.output_binary
    with open(filename, "wb") as f:
        f.write(data)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A small utility to build CUPMB kernels "
        "when underlying CUDA backend does not support template __global__ functions."
    )

    parser.add_argument(
        "--major",
        dest="major",
        type=int,
        default=None,
        help="Device major compute capability.",
    )
    parser.add_argument(
        "--minor",
        dest="minor",
        type=int,
        default=None,
        help="Device minor compute capability.",
    )
    parser.add_argument(
        "--source-file",
        dest="source_file",
        type=str,
        default=None,
        help="Input CUDA source file.",
    )
    parser.add_argument(
        "--input-options",
        dest="input_options",
        type=str,
        default=None,
        help="Source file that defines build options (one per line)",
    )
    parser.add_argument(
        "--input-knames",
        dest="input_knames",
        type=str,
        default=None,
        help="Source file that defines input kernel names (one per line).",
    )
    parser.add_argument(
        "--output-knames",
        dest="output_knames",
        type=str,
        default=None,
        help="Target file that will contain mangled kernel names (one per line).",
    )
    parser.add_argument(
        "--output-binary",
        dest="output_binary",
        type=str,
        default=None,
        help="Target binary file that will contain the binary kernels.",
    )

    args = parser.parse_args()

    assert (args.major is not None) and args.major >= 1, args.major
    assert (args.minor is not None) and args.minor >= 0, args.minor
    assert args.source_file is not None, args.source_file
    assert args.input_options is not None, args.input_options
    assert args.input_knames is not None, args.input_knames
    assert args.output_knames is not None, args.output_knames
    assert args.output_binary is not None, args.output_binary

    if not os.path.isfile(args.source_file):
        msg = f"Source file:///{args.source_file} does not exist."
        parser.error(msg)
    args.source_file = os.path.realpath(args.source_file)

    if not os.path.isfile(args.input_options):
        msg = f"Source file:///{args.input_options} does not exist."
        parser.error(msg)
    args.input_options = os.path.realpath(args.input_options)

    if not os.path.isfile(args.input_knames):
        msg = f"Source file:///{args.input_knames} does not exist."
        parser.error(msg)
    args.input_knames = os.path.realpath(args.input_knames)

    args.kernel_names = _read_lines(args.input_knames)
    args.build_options = _read_lines(args.input_options)

    sys.exit(main(args))
