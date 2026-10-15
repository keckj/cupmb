#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#
"""
On disk cache of the compiled kernels (cubin or ptx and mangled kernel names).

The cache directory is given by the CUPMB_CACHE_DIR environment variable and defaults to
$XDG_CACHE_HOME/cupmb (~/.cache/cupmb). The cache is disabled when CUPMB_DISABLE_KERNEL_CACHE
is set to a non empty value other than 0.

A cache entry is identified by the sha256 of the cupmb version, the backend and its compiler
version, the device compute capability, the kernel source code, the build options, the kernel
names and the content of all the files of the cupmb kernel directory (including the trove headers)
and of the cusvna headers. Headers included from other directories (for instance user headers given
through extra build options) are not part of the key, clear the cache when they change.
"""

import os, json, shutil, hashlib, tempfile, functools

from cupmb.utils.IO import cuda_kernels_dir, cusvna_include_dir

_CACHE_FORMAT_VERSION = 1


def kernel_cache_enabled():
    value = os.environ.get("CUPMB_DISABLE_KERNEL_CACHE", "").strip()
    return value in ("", "0")


def kernel_cache_dir():
    """Directory that contains the compiled kernels (it may not exist yet)."""
    cache_dir = os.environ.get("CUPMB_CACHE_DIR", "").strip()
    if not cache_dir:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip() or os.path.join("~", ".cache")
        cache_dir = os.path.join(xdg_cache_home, "cupmb")
    return os.path.join(os.path.abspath(os.path.expanduser(cache_dir)), "kernels")


def clear_kernel_cache():
    """Remove all the compiled kernels from the on disk cache."""
    cache_dir = kernel_cache_dir()
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)


@functools.cache
def _kernel_headers_digest():
    # computed once per process, symbolic links (trove) are followed
    h = hashlib.sha256()
    for prefix, top in (("", cuda_kernels_dir), ("cusvna", os.path.join(cusvna_include_dir, "cusvna"))):
        for root, dirs, files in os.walk(top, followlinks=True):
            dirs.sort()
            for filename in sorted(files):
                path = os.path.join(root, filename)
                h.update(os.path.join(prefix, os.path.relpath(path, top)).encode())
                h.update(b"\0")
                with open(path, "rb") as f:
                    h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def kernel_cache_key(backend, compiler_version, device_major, device_minor, source_code, kernel_names, build_options):
    from cupmb import __version__

    key = {
        "format": _CACHE_FORMAT_VERSION,
        "cupmb": __version__,
        "backend": backend,
        "compiler": compiler_version,
        "arch": [int(device_major), int(device_minor)],
        "headers": _kernel_headers_digest(),
        "build_options": list(build_options),
        "kernel_names": list(kernel_names),
        "source": hashlib.sha256(source_code.encode()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def _entry_paths(key):
    cache_dir = kernel_cache_dir()
    return os.path.join(cache_dir, f"{key}.bin"), os.path.join(cache_dir, f"{key}.json")


def load_kernel_cache_entry(key):
    """
    Return the path of the cached binary and the mangled kernel names, or None if the entry
    does not exist or is corrupted.
    """
    binary_path, metadata_path = _entry_paths(key)
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        with open(binary_path, "rb") as f:
            binary = f.read()
    except (OSError, ValueError):
        return None
    if metadata.get("sha256") != hashlib.sha256(binary).hexdigest():
        return None
    return binary_path, tuple(metadata["mangled_kernel_names"])


def _atomic_write(path, data, mode):
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_")
    try:
        with os.fdopen(fd, mode) as f:
            f.write(data)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def store_kernel_cache_entry(key, binary, mangled_kernel_names):
    """
    Store a compiled module, return the path of the cached binary. Concurrent writers are
    safe: files are written atomically and the metadata (written last) validates the binary.
    """
    binary_path, metadata_path = _entry_paths(key)
    os.makedirs(os.path.dirname(binary_path), exist_ok=True)
    if isinstance(binary, str):
        binary = binary.encode()
    metadata = {
        "sha256": hashlib.sha256(binary).hexdigest(),
        "mangled_kernel_names": list(mangled_kernel_names),
    }
    _atomic_write(binary_path, binary, "wb")
    _atomic_write(metadata_path, json.dumps(metadata), "w")
    return binary_path
