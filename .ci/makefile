.PHONY: build push

build:
	docker build -t patwie/cuda -f cuda.Dockerfile .
	docker build -t patwie/cpplint -f cpplint.Dockerfile .
	docker build -t patwie/clang-format -f clang.Dockerfile .

