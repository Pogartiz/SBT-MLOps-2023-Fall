import boto3
from botocore.client import Config
from network import Net
import torch
import torch.optim as optim

s3 = boto3.client('s3',
                 endpoint_url='http://localhost:9000',
                 aws_access_key_id='3e6ufA21aGDEEYmmKXG5', 
                aws_secret_access_key='zo20kzKbl46vIDAAF2gQhzQfJLNH3zsbFCS8NjaS')

bucket_name = 'homework'

# Сохранение чекпоинта в бакет

checkpoint_path = 'checkpoints_from_local/model.pt'
#s3.download_file()
s3.upload_file('/home/pogart/SBT-MLOps-2023-Fall/MinIO/checkpoints/model.pt', bucket_name, checkpoint_path)

# Загрузка чекпоинта из бакета
s3.download_file(bucket_name, checkpoint_path, '/home/pogart/SBT-MLOps-2023-Fall/MinIO/checkpoints_from_bucket/model_bucket.pt')

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load('checkpoints_from_bucket/model_bucket.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


