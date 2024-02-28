from azure.storage.blob import BlobServiceClient
from datetime import datetime
import config 

def get_current_timestamp_filename(extension):
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    # append the extension to the timestamp string
    filename = f"{timestamp_str}.{extension}"
    return filename

class BlobUploader:
    def __init__(self, connect_string, container_name):
        self.blob_service_client = BlobServiceClient.from_connection_string(connect_string)
        self.container_name = container_name

    def upload_to_blob(self, file_data, extension):
        file_name = get_current_timestamp_filename(extension)
        blob_client = self.blob_service_client.get_blob_client(self.container_name, file_name)
        blob_client.upload_blob(file_data)
        return blob_client.url

