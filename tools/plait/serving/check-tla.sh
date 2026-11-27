#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

state_dir=$(mktemp -d "${TMPDIR:-/tmp}/vllm-plait-serving-tlc.XXXXXXXX")
trap 'rm -rf -- "$state_dir"' EXIT

tlc_command=(tlc)
if [[ -n ${TLA2TOOLS_JAR:-} ]]; then
    tlc_command=(java -cp "$TLA2TOOLS_JAR" tlc2.TLC)
fi

check() {
    "${tlc_command[@]}" -deadlock -metadir "$state_dir/$1" -config "specs/$1.cfg" "specs/$2.tla"
}

check PlaitServingIngress PlaitServingIngress
check PlaitNativeServingIngress PlaitServingIngress
check PlaitCompletionReplay PlaitCompletionReplay
check PlaitCompletionKvResume PlaitCompletionReplay
check PlaitCompletionPrefixResume PlaitCompletionPrefixResume
check PlaitShardedPromptScores PlaitShardedPromptScores
check PlaitShardedPromptScoresRows PlaitShardedPromptScores
check PlaitShardedPromptOverlap PlaitShardedPromptOverlap
check PlaitPlannedStreamHandoff PlaitPlannedStreamHandoff
check PlaitSeededStreamHandoff PlaitPlannedStreamHandoff
check PlaitKvHandoff PlaitKvHandoff
check PlaitPdStreamReplay PlaitPdStreamReplay
check PlaitServingModePolicy PlaitServingModePolicy
check PlaitServingProvision PlaitServingProvision
check PlaitServingCalibration PlaitServingCalibration
check PlaitServingProfileRestore PlaitServingProfileRestore
