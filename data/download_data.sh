#!/bin/bash

# NOTE: If it failed to auto get file, please visit link to manual download
#       https://drive.google.com/file/d/1IXTnxG1P4q79732NR-d3a2wqWGE68M1w/view

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IXTnxG1P4q79732NR-d3a2wqWGE68M1w' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IXTnxG1P4q79732NR-d3a2wqWGE68M1w" -O dataset_raw.zip && rm -rf /tmp/cookies.txt

unzip dataset_raw.zip
