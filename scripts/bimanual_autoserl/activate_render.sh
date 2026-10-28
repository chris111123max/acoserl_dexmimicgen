# Prefer installed system OSMesa. Optional override for a user-owned library directory.
if [[ -n "${ACOSERL_OSMESA_LIBDIR:-}" ]]; then
 export LD_LIBRARY_PATH="$ACOSERL_OSMESA_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
