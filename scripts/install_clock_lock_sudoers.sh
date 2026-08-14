#!/bin/bash
# One-time setup so every run can pin the GPU core clock without a password.
#
#   sudo bash scripts/install_clock_lock_sudoers.sh
#
# Installs two things:
#   /usr/local/sbin/kernelmem-gpu-clock   root-owned wrapper, strict arg grammar
#   /etc/sudoers.d/kernelmem-clock-lock   NOPASSWD for exactly that wrapper
#
# The sudoers rule names ONLY the wrapper. It deliberately does not grant
# nvidia-smi: sudo's wildcards match spaces, so any rule permissive enough to
# cover `-lgc <mhz>` also covers `-f /etc/sudoers`, which nvidia-smi will happily
# write to as root. The wrapper validates its own arguments instead, so granting
# it with unrestricted args is still tightly bounded.
#
# To undo:
#   sudo rm -f /etc/sudoers.d/kernelmem-clock-lock /usr/local/sbin/kernelmem-gpu-clock
set -euo pipefail

WRAPPER_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kernelmem-gpu-clock"
WRAPPER_DST=/usr/local/sbin/kernelmem-gpu-clock
SUDOERS_DST=/etc/sudoers.d/kernelmem-clock-lock

if [[ $EUID -ne 0 ]]; then
  echo "This installs a sudoers rule, so it must run as root:" >&2
  echo "    sudo bash $0" >&2
  exit 1
fi

# The user that will RUN the benchmarks -- the invoking user, not root.
TARGET_USER=${SUDO_USER:-${1:-}}
if [[ -z "$TARGET_USER" || "$TARGET_USER" == "root" ]]; then
  echo "Could not determine the benchmarking user. Pass it explicitly:" >&2
  echo "    sudo bash $0 <username>" >&2
  exit 1
fi
id -u "$TARGET_USER" >/dev/null 2>&1 || { echo "no such user: $TARGET_USER" >&2; exit 1; }

[[ -f "$WRAPPER_SRC" ]] || { echo "missing wrapper source: $WRAPPER_SRC" >&2; exit 1; }

install -o root -g root -m 0755 "$WRAPPER_SRC" "$WRAPPER_DST"
echo "installed $WRAPPER_DST"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
cat >"$TMP" <<EOF
# Installed by KernelMem scripts/install_clock_lock_sudoers.sh
# Lets $TARGET_USER pin/release the GPU core clock so benchmark runs are
# reproducible. The wrapper accepts only: lock <idx> <mhz> | unlock <idx> |
# status <idx>, with both numbers validated, and cannot run anything else.
$TARGET_USER ALL=(root) NOPASSWD: $WRAPPER_DST
EOF

# Never install a sudoers file that does not parse -- a broken one can lock the
# machine out of sudo entirely.
if ! visudo -cqf "$TMP"; then
  echo "generated sudoers file failed validation; nothing was installed" >&2
  exit 1
fi
install -o root -g root -m 0440 "$TMP" "$SUDOERS_DST"
echo "installed $SUDOERS_DST"

echo
echo "Verifying as $TARGET_USER ..."
if sudo -u "$TARGET_USER" sudo -n "$WRAPPER_DST" status 0 >/dev/null 2>&1; then
  echo "OK -- clock locking is available. Check it with:"
  echo "    python -m utils.clock_lock --status"
else
  echo "WARNING: the wrapper still is not callable without a password." >&2
  echo "Check for a later rule in /etc/sudoers overriding sudoers.d." >&2
  exit 1
fi
