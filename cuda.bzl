def _cuda_binary(ctx):
  default_flags = ctx.attr.flags + \
      " -x=cu -Xcompiler \"-O3 -Ofast -Wall -Wextra -DWITH_CUDA\""

  cmd = "/usr/local/cuda/bin/nvcc -D__CUDACC__ "
  cmd += " " + default_flags + " "

  for src in ctx.files.srcs:
    cmd += src.path

  executable = ctx.actions.declare_file(ctx.attr.name)
  cmd += " -o " + executable.path

  for include in ctx.attr.includes:
    cmd += " -I" + include

  ctx.actions.run_shell(
      outputs=[ctx.actions.declare_file(ctx.label.name)],
      inputs=ctx.files.srcs + ctx.files.hdrs,
      command=cmd,
      mnemonic="CudaCompile",
      progress_message="compile cuda",
      use_default_shell_env=True,
  )

  return [DefaultInfo(
      files=depset([executable]),
      executable=executable,
  )]


cuda_binary = rule(
    implementation=_cuda_binary,
    executable=True,
    attrs={
        "flags": attr.string(default=""),
        "srcs": attr.label_list(default=[], allow_files=[".cc"]),
        "hdrs": attr.label_list(default=[], allow_files=[".h"]),
        "includes": attr.string_list(default=[]),
        "out": attr.output(mandatory=False),
    },
)
