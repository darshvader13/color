import boto3
import torch
import io
import base64

def lambda_handler(event, context):
    bucket_name = 'colorizemanga'
    generator_key = 'ColorizeArtistic_gen.pth'
    finetuned_key = 'finetuned_default_generator.pth'

    try:

        if 'image' in event:
            image_data = event['image']
            image_bytes = base64.b64decode(image_data)
            with open('/tmp/image.jpg', 'wb') as f:
                f.write(image_bytes)
            print('Image saved to /tmp/image.jpg')

        s3 = boto3.client('s3')
        generator_response = s3.get_object(Bucket=bucket_name, Key=generator_key)
        generator_data = generator_response['Body'].read()
        generator_model = torch.load(io.BytesIO(generator_data))
        print(f'Downloaded object: {generator_key}')

        finetuned_response = s3.get_object(Bucket=bucket_name, Key=finetuned_key)
        finetuned_data = finetuned_response['Body'].read()
        finetuned_model = torch.load(io.BytesIO(finetuned_data))
        print(f'Downloaded object: {finetuned_key}')

    except Exception as e:
        print(e)
        raise e
    
