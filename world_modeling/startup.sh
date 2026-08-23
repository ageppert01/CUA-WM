#!/bin/bash
# startup.sh — unified container entry point for CUA-WM jobs.
#
# Usage:
#     ./startup.sh <mode>
#
# Modes:
#     baseline                            Vanilla OpenCUA API (opencua_api.py)
#     framework                           framework_api.py with current defaults
#     framework-no-greedy-after-code      framework_api.py --no-greedy-after-code
#     smoke-test                          smoke_test_framework.py with current defaults
#     smoke-test-no-greedy-after-code     smoke_test_framework.py --no-greedy-after-code
#
# Modes describe which CLI flags are passed; defaults are defined inside
# framework_api.py and smoke_test_framework.py. Adding a new ablation
# axis = one new mode line below.
#
# HTCondor submit files should point `executable = startup.sh` and
# `arguments = <mode>`.

MODE="${1:-}"
if [ -z "${MODE}" ]; then
    echo "Usage: $0 <mode>"
    echo "Modes:"
    echo "  baseline                            Vanilla OpenCUA API"
    echo "  framework                           Framework API, current defaults"
    echo "  framework-no-greedy-after-code      Framework API, --no-greedy-after-code"
    echo "  smoke-test                          Smoke test, current defaults"
    echo "  smoke-test-no-greedy-after-code     Smoke test, --no-greedy-after-code"
    exit 1
fi

# shellcheck source=common.sh
. ./common.sh

prepare_model_workspace

FRAMEWORK_DEPS=("peft>=0.12,<0.14" "huggingface_hub>=0.25,<1.0")
SMOKE_DEPS=("peft>=0.12,<0.14" "huggingface_hub>=0.25,<1.0" "Pillow")

run_framework() {
    install_packages_with_strip "${FRAMEWORK_DEPS[@]}"
    setup_ssh_key
    section "Starting framework API"
    python framework_api.py --n-candidates 2 "$@" > api.log 2>&1 &
    open_ssh_tunnel
}

run_smoke_test() {
    install_packages_with_strip "${SMOKE_DEPS[@]}"
    section "Running smoke test"
    python smoke_test_framework.py "$@"
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "RESULT: SOME TESTS FAILED"
        exit 1
    fi
    echo "RESULT: ALL TESTS PASSED"
}

case "${MODE}" in
    baseline)
        setup_ssh_key
        section "Starting baseline API (opencua_api.py)"
        python opencua_api.py > api.log 2>&1 &
        open_ssh_tunnel
        ;;

    framework)
        run_framework
        ;;

    framework-no-greedy-after-code)
        run_framework --no-greedy-after-code
        ;;

    smoke-test)
        run_smoke_test
        ;;

    smoke-test-no-greedy-after-code)
        run_smoke_test --no-greedy-after-code
        ;;

    *)
        echo "Unknown mode: ${MODE}"
        echo "Run with no arguments to see valid modes."
        exit 1
        ;;
esac