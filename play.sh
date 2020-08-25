#!/usr/bin/env bash

ssh -i ~/Workspace/applesay applesay@139.224.19.230
ssh -i ~/Workspace/new_ai_box root@139.224.19.230

docker system prune -a -f

docker image build -t joshuaze/img2tags:3.0 .

docker push joshuaze/img2tags:3.0

docker run -m 1000m -p 8081:8081 joshuaze/img2tags:3.0

docker-compose --project-name img2tags up -d
docker-compose --project-name img2tags down

docker stats
