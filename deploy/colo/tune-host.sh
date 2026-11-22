#!/usr/bin/env bash
# Host tuning that sysctl can't express: CPU governor/C-states, IRQ affinity,
# huge pages. Intended to run once at boot as a systemd oneshot unit, AFTER the
# kernel cmdline changes in firewall-node-a.grub.example have taken effect
# (isolcpus etc. need a reboot; this script does not).
#
# Every value marked ADAPT depends on the real core layout for this VM -- get
# it from the provider / your OpenNebula VM template, don't guess.
set -euo pipefail

# ADAPT: cores pinned to this application (see docker-compose.colo.yml cpuset
# values) -- these are EXCLUDED from IRQ affinity and get the performance
# governor. Cores NOT in this list are left for the OS, housekeeping and NIC
# interrupts.
APP_CORES=(2 3 4 5)

log() { echo "[tune-host] $*"; }

require_root() {
  if [[ $EUID -ne 0 ]]; then
    echo "must run as root" >&2
    exit 1
  fi
}

cpu_governor_performance() {
  # Removes frequency-scaling transitions (and their latency) on the
  # application's cores. Requires the intel_pstate/acpi-cpufreq driver to
  # expose the sysfs governor interface; on some cloud/KVM guests frequency
  # scaling is not exposed to the guest at all, which is fine -- it just means
  # the host already fixed it (ask the provider to confirm).
  for c in "${APP_CORES[@]}"; do
    path="/sys/devices/system/cpu/cpu${c}/cpufreq/scaling_governor"
    if [[ -w "$path" ]]; then
      echo performance > "$path"
      log "cpu${c}: governor -> performance"
    else
      log "cpu${c}: no cpufreq sysfs entry (likely fixed by the hypervisor host -- confirm with the provider)"
    fi
  done
}

disable_c_states() {
  # Deep C-states save power at the cost of a wake-up latency spike -- the
  # kind of jitter this whole deployment is trying to remove. Requires
  # `cpupower` (linux-tools); skips quietly if C-state control isn't exposed
  # to the guest (common on KVM/OpenNebula VPS without CPU passthrough).
  if command -v cpupower >/dev/null 2>&1; then
    for c in "${APP_CORES[@]}"; do
      cpupower -c "$c" idle-set -D 0 2>/dev/null \
        && log "cpu${c}: deep C-states disabled" \
        || log "cpu${c}: C-state control not available to this guest"
    done
  else
    log "cpupower not installed; skipping C-state tuning"
  fi
}

irq_affinity_away_from_app_cores() {
  # Steer every IRQ onto the cores NOT used by the pinned application, so NIC
  # interrupts never preempt the firewall/model processes. This is a blunt
  # instrument (it moves ALL IRQs, not just the NIC's); for a single-purpose
  # node that is usually the right default -- narrow it to specific IRQ
  # numbers (`grep <nic-name> /proc/interrupts`) if this node has other
  # latency-sensitive interrupt sources.
  total_cores=$(nproc)
  other_mask=0
  for ((c = 0; c < total_cores; c++)); do
    skip=0
    for a in "${APP_CORES[@]}"; do [[ "$c" == "$a" ]] && skip=1; done
    [[ $skip -eq 0 ]] && other_mask=$((other_mask | (1 << c)))
  done
  if [[ $other_mask -eq 0 ]]; then
    log "no non-application cores to steer IRQs to -- leaving affinity untouched"
    return
  fi
  mask_hex=$(printf '%x' "$other_mask")
  moved=0
  for irq_dir in /proc/irq/*/; do
    irq=$(basename "$irq_dir")
    [[ "$irq" =~ ^[0-9]+$ ]] || continue
    affinity_file="${irq_dir}smp_affinity"
    # Some IRQs (timers, IPIs) don't accept affinity changes even as root;
    # -w also makes this safe to dry-run without root.
    if [[ -w "$affinity_file" ]]; then
      echo "$mask_hex" > "$affinity_file" && moved=$((moved + 1))
    fi
  done
  log "steered ${moved} IRQ(s) to cpu mask 0x${mask_hex} (cores outside ${APP_CORES[*]})"
}

huge_pages_madvise() {
  # See sysctl-colo.conf for why "madvise" rather than "always": explicit opt-in
  # for the ONNX Runtime / torch processes, no background compaction stalls on
  # processes that didn't ask for huge pages.
  if [[ -w /sys/kernel/mm/transparent_hugepage/enabled ]]; then
    echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
    log "transparent huge pages -> madvise"
  fi
  # ADAPT: explicit huge-page pool size for the model-serving processes, e.g.
  #   echo 1024 > /proc/sys/vm/nr_hugepages     # 1024 x 2 MB = 2 GB
  # left commented out: sizing depends on model memory footprint.
}

main() {
  require_root
  cpu_governor_performance
  disable_c_states
  irq_affinity_away_from_app_cores
  huge_pages_madvise
  log "done"
}

main "$@"
