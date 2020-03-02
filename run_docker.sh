#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

if [[ -z ${GPUS} ]]; then
    GPUS='all'
fi

docker run --gpus ${GPUS} -it --rm --network host --ipc=host \
  --mount src=$(pwd),target=/root/code/,type=bind \
  ntutselab/tf_image:latest \
  bash -c "cd /root/code/ && $cmd_line"
