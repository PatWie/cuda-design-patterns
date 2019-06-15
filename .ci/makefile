.PHONY: build push

build:
	docker build --build-arg VERCUDA=10.1 -t patwie/cuda:10.1 -f cuda.Dockerfile .
	docker build --build-arg VERCUDA=10.0 -t patwie/cuda:10.0 -f cuda.Dockerfile .
	docker build --build-arg VERCUDA=9.1 -t patwie/cuda:9.1 -f cuda.Dockerfile .
	docker build --build-arg VERCUDA=9.0 -t patwie/cuda:9.0 -f cuda.Dockerfile .
	docker build -t patwie/cpplint -f cpplint.Dockerfile .
	docker build -t patwie/clang-format -f clang.Dockerfile .

