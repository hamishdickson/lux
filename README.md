# lux ai 2021 solution

RL Kaggle comp https://www.kaggle.com/c/lux-ai-2021

Solution heavily based on HandyRL https://github.com/DeNA/HandyRL

note: switch to https://github.com/glmcdona/LuxPythonEnvGym, it's 45 times quicker

viewer https://2021vis.lux-ai.org/


## installing everything (ubuntu 20.04 LTS)

bit of a general pain. You need node v12+

https://stackoverflow.com/questions/63312642/how-to-install-node-tar-xz-file-in-linux

then add to path (I use fish)

set -U fish_user_paths /usr/node-v16.13.0-linux-x64

change permissions, some version of this

sudo chmod a+w /usr/node-v16.13.0-linux-x64/lib/node_modules/ 

## install lux ai

npm install -g @lux-ai/2021-challenge@latest

## install kaggle envs

pip install kaggle-environments -U

