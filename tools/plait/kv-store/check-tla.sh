#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

state_dir=$(mktemp -d "${TMPDIR:-/tmp}/vllm-plait-kv-tlc.XXXXXXXX")
trap 'rm -rf -- "$state_dir"' EXIT

tlc_command=(tlc)
if [[ -n ${TLA2TOOLS_JAR:-} ]]; then
    tlc_command=(java -cp "$TLA2TOOLS_JAR" tlc2.TLC)
fi

check() {
    "${tlc_command[@]}" -deadlock -metadir "$state_dir/$1" -config "specs/$1.cfg" "specs/$2.tla"
}

check PlaitCheckpointAdmission PlaitCheckpointAdmission
check PlaitCheckpointPeers PlaitCheckpointPeers
check PlaitCheckpointPeerPlan PlaitCheckpointPeerPlan
check PlaitCheckpointBudgetSwitch PlaitCheckpointBudgetSwitch
check PlaitCheckpointBudgetSwitchThree PlaitCheckpointBudgetSwitch
check PlaitKvPublication PlaitKvPublication
check PlaitLayeredKv PlaitLayeredKv
