# Decision

Keep the consolidated current-head source as an unpromoted candidate. The new
native image passes bounded GPU smoke, but its guarded three-rank start reached
the 5 GiB memory floor after checkpoint loading on dgx3. The known working
patch0029 service has been restored on all three nodes. Do not lower the guard
or treat this image as qualified.

The omitted managed-weight allocator may matter on GB10. First validate the
smaller B12X lifecycle fix: current vLLM no longer calls `_log_loading_time`,
so the loader must finish queued transfers and release cached CUDA allocations
before post-load weight preparation. If a guarded restart still lacks headroom,
measure the allocator delta and restore a focused managed-weight path only if
needed. Output quality, long-context capacity, and serving speed remain open
gates on the consolidated source.
