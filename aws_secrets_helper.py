import boto3
from botocore.exceptions import ClientError

def get_secrets(secret_name="example", region_name="us-east-2"):
    """
    Fetch secrets from AWS Secrets Manager.

    Returns:
        dict: A dictionary containing aws_secret_id, aws_access, and openai_api_key.
    """
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secrets = eval(get_secret_value_response['SecretString'])

    # Extract specific keys from the secrets
    aws_secret_id = secrets.get('aws_secret_id')
    aws_access = secrets.get('aws_access')
    openai_api_key = secrets.get('openai_api_key')

    return {
        "aws_secret_id": aws_secret_id,
        "aws_access": aws_access,
        "openai_api_key": openai_api_key
    }