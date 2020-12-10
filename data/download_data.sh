#!/bin/bash

# NOTE: If it failed to auto get file, please visit link to manual download
#       https://drive.google.com/u/0/uc?id=18WbAzh66uNLBX-zuejBjFcXP9qNb0HZF&export=download

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18WbAzh66uNLBX-zuejBjFcXP9qNb0HZF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18WbAzh66uNLBX-zuejBjFcXP9qNb0HZF" -O dataset_raw.zip && rm -rf /tmp/cookies.txt
unzip dataset_raw.zip
