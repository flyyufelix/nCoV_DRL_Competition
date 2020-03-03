IMAGE_NAME="virus_rl"
IMAGE_VERSION=0.1

# Build Image
docker build -f Dockerfile -t $IMAGE_NAME:$IMAGE_VERSION .

# Run container and execute rl trainer
docker run -p 8888:8888 $IMAGE_NAME:$IMAGE_VERSION jupyter notebook rl_trainer.ipynb --ip 0.0.0.0 --no-browser --allow-root


