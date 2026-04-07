import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

class DriveHandler:
    def __init__(self, credentials_path):
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']
        self.creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=self.scopes)
        self.service = build('drive', 'v3', credentials=self.creds)

    def _extract_id(self, folder_input):
        """Extracts ID from URL if necessary."""
        if "drive.google.com" in folder_input:
            # Handles https://drive.google.com/drive/folders/ID?usp=sharing
            parts = folder_input.split('/')
            last_part = parts[-1].split('?')[0]
            # Sometimes it's the second to last part if there's a trailing slash
            if not last_part and len(parts) > 1:
                last_part = parts[-2].split('?')[0]
            return last_part
        return folder_input

    def list_files_in_folder(self, folder_id):
        folder_id = self._extract_id(folder_id)
        query = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false"
        results = self.service.files().list(
            q=query, fields="nextPageToken, files(id, name)").execute()
        return results.get('files', [])

    def download_file(self, file_id, dest_path):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        with open(dest_path, 'wb') as f:
            f.write(fh.getvalue())
        return dest_path

    def download_folder(self, folder_id, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        files = self.list_files_in_folder(folder_id)
        downloaded_paths = []
        for f in files:
            dest = os.path.join(local_dir, f['name'])
            self.download_file(f['id'], dest)
            downloaded_paths.append(dest)
        return downloaded_paths
