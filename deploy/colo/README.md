# Colocated / low-latency deployment — starting point

Companion to [`docs/infrastructure/colo-deployment.md`](../../docs/infrastructure/colo-deployment.md).
These are a first pass at turning that design into an actual VM image and
compose file, **not** a validated tuning profile — every setting here depends
on the real NIC, kernel and hypervisor you get from the provider. Treat
anything marked `# ADAPT` as something that needs a value from your hardware,
not a default to trust.

| file | what it does | apply at |
|---|---|---|
| `docker-compose.colo.yml` | host networking, CPU pinning, real-time scheduling + memlock capabilities for the hot-path containers | `docker compose -f ../../docker-compose.yml -f docker-compose.colo.yml up` |
| `sysctl-colo.conf` | host-level network/NUMA/memory tuning | `sysctl --system` after copying into `/etc/sysctl.d/`, or via cloud-init |
| `tune-host.sh` | CPU governor, C-states, IRQ affinity, huge pages — the things sysctl can't set | run once at boot (systemd oneshot unit), **after** reading it and adjusting the `# ADAPT` values |
| `firewall-node-a.grub.example` | kernel command-line additions (`isolcpus`, `nohz_full`, hugepages) | bake into the VM image; requires a reboot |

## Order of operations

1. Get the real topology from the provider (§10 of the design doc): which
   cores are pinned to this VM, which NUMA node, whether SR-IOV is available,
   whether there's a PTP feed.
2. Fill in the `# ADAPT` values in every file below with that topology.
3. Bake the kernel cmdline + `tune-host.sh` into the VM image (or apply once
   via cloud-init + a reboot) **before** bringing up the application.
4. Bring the application up with the compose override.
5. Verify: `cat /sys/devices/system/cpu/cpu<N>/cpufreq/scaling_governor`,
   `cat /proc/interrupts` (NIC IRQs should not land on the pinned cores),
   `chronyc sourcestats` or `pmc -u -i <iface>` for PTP sync status.
6. Only then run a real load test against this node — not a shared sandbox —
   and compare it against `bench/controlled/REPORT.md`'s *shape* (which stage
   costs what, relative to the others), not its absolute numbers.
