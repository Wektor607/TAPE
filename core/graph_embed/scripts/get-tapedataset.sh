#!/usr/bin/env bash
# LINK=https://drive.google.com/file/d/1ueqmPH2L5flLQjTIgHxLO2-FlGMvyCqy/view?usp=sharing
# If automatic downloading isn't working, the file can be downloaded manually with the above link.
ggID='1ueqmPH2L5flLQjTIgHxLO2-FlGMvyCqy'
ggURL='https://drive.google.com/uc?export=download'
TARGET=dataset.tar.gz
SHA256SUM=ca63ec67f4be6a03433dad1b7ee6b0d122d8bc57da219d7ac8e59b481bd9b7ff

cd "$(dirname ${BASH_SOURCE[0]})/.."

# # Automatic downloading script adopted from https://stackoverflow.com/a/38937732
# filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
# getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
# curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${TARGET}"

if ! command -v gdown &> /dev/null
then
    read -p "Prerequisite package gdown is not installed. Press any key to install (pip install --upgrade gdown)"
    pip install --upgrade gdown
fi
gdown -O "${TARGET}" ${ggID}

echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r

# please run these commands in the terminal to unzip and clean the dataset
tar -xvzf $TARGET