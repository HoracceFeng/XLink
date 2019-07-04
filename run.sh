docker run -it $1 \
	-v /Users/horacce/Nirva/Data/STR:/data \
	-v /Users/horacce/Nirva/Project/STR-OCR:/code \
	10.202.56.200:5000/ubuntu16_cuda8_cudnn6_py2_sklearn_cv3.4:pixel_link_cpu bash
