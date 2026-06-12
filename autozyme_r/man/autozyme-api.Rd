\name{autozyme-api}
\alias{activate}
\alias{auto_threads}
\alias{blas_info}
\alias{dashboard}
\alias{env_snapshot}
\alias{inject_all}
\alias{inspect}
\alias{is_disabled}
\alias{list_patches}
\alias{list_subsets}
\alias{register_patch}
\alias{resolve_dataset_path}
\alias{restore}
\alias{restore_all}
\alias{set_threads}
\alias{speedups}
\alias{status}
\alias{subset_patches}
\alias{verify_patch}
\alias{with_disabled}
\title{AutoZyme patch registry API}
\description{
User-facing entry points for registering, activating, restoring, inspecting,
and verifying AutoZyme R patches.
}
\details{
This shared manual page documents the exported AutoZyme API surface. The
functions operate on the package patch registry, thread controls, packaged
speedup snapshots, and verification helpers.
}
\keyword{package}
