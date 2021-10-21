#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, glob, shutil
import distutils, distutils.command.build

script_dir = os.path.dirname(os.path.realpath(__file__))
module_dir = os.path.join(script_dir, "cupmb")
kernel_dir = os.path.join(module_dir, "kernels")

package_license = os.path.join(script_dir, "LICENSE")
trove_license = os.path.join(script_dir, "ext", "trove", "LICENSE")
if not os.path.isfile(trove_license):
    msg = (
        "Trove dependency is missing, please initialize git submodules: \n"
        "  git submodule update --init --recursive"
    )
    raise RuntimeError(msg)


class BuildCommand(distutils.command.build.build):
    def run(self):
        distutils.command.build.build.run(self)
        target_dir = os.path.join(self.build_lib, "cupmb")
        shutil.copyfile(package_license, os.path.join(target_dir, "LICENSE"))
        shutil.copyfile(
            trove_license, os.path.join(target_dir, "kernels", "trove", "LICENSE")
        )


data_files = []
data_files += glob.glob(os.path.join(kernel_dir, "*.cuh"), recursive=True)
data_files += glob.glob(os.path.join(kernel_dir, "trove", "**", "*.h"), recursive=True)

data_files = [os.path.relpath(path, module_dir) for path in data_files]

with open(os.path.join(script_dir, "README.md")) as f:
    long_description = f.read()

distutils.core.setup(
    name="cupmb",
    version="2021.10.dev0",
    description="Cuda Parallel Marching Blocks, supports both pycuda and cupy backends.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jean-Baptiste Keck",
    author_email="jbkeck@hotmail.com",
    packages=["cupmb", "cupmb.backend", "cupmb.utils"],
    package_data={"": data_files},
    cmdclass={"build": BuildCommand},
)
