IMAGE_NAME="virus_rl"
IMAGE_VERSION=0.1

# Build Image
docker build -f Dockerfile -t $IMAGE_NAME:$IMAGE_VERSION .

# Run container and execute rl trainer
#docker run --runtime=nvidia $IMAGE_NAME:$IMAGE_VERSION python run_rl.py
docker run $IMAGE_NAME:$IMAGE_VERSION python run_rl.py


