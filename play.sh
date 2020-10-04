#!/usr/bin/env bash

docker system prune -a -f

docker image build -t joshuaze/img2tags:3.1 .

docker push joshuaze/img2tags:3.1

docker run -m 1000m -p 8081:8081 joshuaze/img2tags:3.1

ssh -i ~/Workspace/applesay applesay@139.224.19.230
ssh -i ~/Workspace/new_ai_box root@139.224.19.230

scp -i ~/Workspace/applesay -r ./models applesay@139.224.19.230:/home/applesay/image2tags
scp -i ~/Workspace/applesay -r ./docker-compose.yml applesay@139.224.19.230:/home/applesay/image2tags

docker-compose down
docker-compose up -d
docker stats
