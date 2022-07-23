# This script is only suitable for Linux Environment
# If you are using conda env, please activate your env first, then run this script
# Author: ZHCSOFT

pythonBin=$(dirname `which python`)
pythonBin=$(dirname $pythonBin)
pythonLibBin=$pythonBin"/lib"
export LD_LIBRARY_PATH=$pythonLibBin":$LD_LIBRARY_PATH"
echo $LD_LIBRARY_PATH
python InferOpenVINO.py