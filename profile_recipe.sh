#!/usr/bin/env bash
recipe="$1"
output=profile_$(basename -s .yml "$recipe").json
py-spy record \
--idle \
--rate 10 \
--subprocesses \
--format speedscope \
--output "$output" \
esmvaltool run "$recipe"
echo "Profiling information written to $output"
