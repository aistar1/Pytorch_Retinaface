#!/bin/bash

project_name=retinaface
tag=20.08

docker_name=${project_name}:${tag}
docker build -t ${docker_name} -f Dockerfile
docker-compose run --rm retinaface
