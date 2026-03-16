#!/usr/bin/env bash
set -euo pipefail

root_dir="$(git rev-parse --show-toplevel)"
cd "${root_dir}"

starcraft_path="${root_dir}/deps/StarCraftII"
map_dir="${starcraft_path}/Maps"
printf "* Installing StarCraft 2 and SMACv2 maps into ${starcraft_path}\n"

# Extract SC2 into deps/ so it lands at deps/StarCraftII
cd "${root_dir}/deps"
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
# The archive contains StarCraftII folder. Password comes from the documentation.
unzip -qo -P iagreetotheeula SC2.4.10.zip
rm SC2.4.10.zip

mkdir -p "${map_dir}/SMAC_Maps"
# Install Maps
cd "${root_dir}"
wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv *.SC2Map "${map_dir}/SMAC_Maps"
rm SMAC_Maps.zip

printf "StarCraft II and SMAC maps are installed.\n"
printf "Run 'pip install -e ./deps/smacv2' to install the smacv2 Python package.\n"
