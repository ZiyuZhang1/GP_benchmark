#!/bin/bash
# This script has been renamed to run_one_disease.sh.
# Usage: ./src/run_one_disease.sh <ICD10_DISEASE_ID>
exec "$(dirname "${BASH_SOURCE[0]}")/run_one_disease.sh" "$@"
