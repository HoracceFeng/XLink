set -x
set -e

export CUDA_VISIBLE_DEVICES=$1

python test_portable.py \
            --checkpoint_path=$2 \
            --dataset_dir=$3 \
            --output_dir=$4 \
            --draw_image=$5 \
            --scale_resize=$6 \
            --pixel_conf_threshold=0.5 \
            --link_conf_threshold=0.5 \
            --gpu_memory_fraction=-1
