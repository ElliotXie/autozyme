from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


SRC = Path("src") / "autozyme" / "_native_scanpy_src"

extra_compile_args = []
extra_link_args = []
libraries = []

if os.name == "posix":
    extra_compile_args.append("-std=c99")
    if sys.platform != "darwin":
        extra_compile_args.append("-pthread")
        extra_link_args.append("-pthread")
elif sys.platform == "win32":
    # MSVC: select C11 so the kernels' C99 compound literals / designated
    # initializers compile. Threading uses Win32 CRITICAL_SECTION /
    # CONDITION_VARIABLE via az_pthread_compat.h (no winpthreads needed); those
    # live in kernel32, which MSVC links implicitly, so no extra libs/flags.
    extra_compile_args.append("/std:c11")

ext_modules = [
    Extension(
        "autozyme._native_scanpy",
        sources=[
            str(SRC / "module.c"),
            str(SRC / "knn.c"),
            str(SRC / "umap_graph.c"),
            str(SRC / "umap_layout.c"),
            str(SRC / "umap_layout_parallel.c"),
        ],
        include_dirs=[str(SRC)],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]


class _OptionalBuildExt(_build_ext):
    """Never fail the whole install if the native accelerator won't build.

    The scanpy patches transparently fall back to stock scanpy when the
    ``_native_scanpy`` extension is absent, so a build failure is a performance
    regression, not a broken install. This keeps ``pip install autozyme``
    working on any toolchain where the kernel won't compile -- notably
    Windows/MSVC, whose build path is wired here but not yet CI-validated.
    """

    def run(self):
        try:
            super().run()
        except Exception as exc:  # noqa: BLE001 - any build failure is non-fatal
            self._skip(exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001
            self._skip(exc)

    @staticmethod
    def _skip(exc):
        sys.stderr.write(
            "\n[autozyme] WARNING: native scanpy accelerator did not build "
            f"({exc!r}); falling back to stock scanpy at runtime.\n"
        )


setup(ext_modules=ext_modules, cmdclass={"build_ext": _OptionalBuildExt})
