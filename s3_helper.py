import boto3

class S3Helper:
    def __init__(self, bucket_name):
        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket(bucket_name)

    def upload_file(self, file_name, object_name=None):
        if object_name is None:
            object_name = file_name
        self.bucket.upload_file(file_name, object_name)

    def download_file(self, object_name, file_name=None):
        if file_name is None:
            file_name = object_name
        self.bucket.download_file(object_name, file_name)