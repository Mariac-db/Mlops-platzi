sudo apt update

sudo apt install apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"

sudo apt update

apt-cache policy docker-ce

sudo apt install docker-ce

sudo apt install docker-compose

# Optional to run docker without sudo
sudo usermod -aG docker ${USER}

sudo su - ${USER}

id -nG

sudo usermod -aG docker ubuntu

## END ##

docker ps

      - POSTGRES_USER=fastapi_traefik_prod
      - POSTGRES_PASSWORD=fastapi_traefik_prod
      - POSTGRES_DB=fastapi_traefik_prod
