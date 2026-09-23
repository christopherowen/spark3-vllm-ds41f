# Decision

Keep the consolidated current-head source as an unpromoted candidate. The new
native image passes bounded GPU smoke, but its guarded three-rank start reached
the 5 GiB memory floor after checkpoint loading on dgx3. The known working
patch0029 service has been restored on all three nodes. Do not lower the guard
or treat this image as qualified.

The smaller B12X lifecycle fix did not recover memory headroom. The latest
guarded load repeated the post-weight failure at 5,164,244 KiB available on
dgx3. Patch 0005 now restores managed storage for final model weights on GB10
as one distinct concept, retaining upstream's bounded reader. Test a small
native checkpoint copy in an 8 GiB container before the next full launch.
Output quality, long-context capacity, and serving speed remain open gates.
