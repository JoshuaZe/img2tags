#!/usr/bin/env bash

docker image build -t joshuaze/img2tags:1.0 .

docker push joshuaze/img2tags:1.1

docker run -m 1000m -p 8081:8081 joshuaze/img2tags:1.0

docker-compose --project-name img2tags up -d
docker-compose --project-name img2tags down

docker stats
