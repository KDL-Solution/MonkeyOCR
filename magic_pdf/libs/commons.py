def join_path(
    *args,
):
    return "/".join(str(s).rstrip("/") for s in args)
