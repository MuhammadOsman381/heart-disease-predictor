from pathlib import Path

from supabase import Client, create_client

from app.config import get_settings


class ArtifactStorage:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: Client | None = None
        if self.settings.supabase_url and self.settings.supabase_key:
            self.client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_key,
            )

    def upload_file(self, local_path: Path, remote_path: str) -> str:
        if not self.client:
            return str(local_path)

        bucket = self.client.storage.from_(self.settings.supabase_bucket)
        with local_path.open("rb") as file_obj:
            bucket.upload(
                path=remote_path,
                file=file_obj,
                file_options={"content-type": "application/octet-stream", "upsert": "true"},
            )
        return remote_path

    def download_to_cache(self, remote_path: str) -> Path:
        local_path = Path(remote_path)
        if local_path.exists():
            return local_path

        if not self.client:
            raise RuntimeError("Supabase credentials are missing and artifact is not local.")

        data = self.client.storage.from_(self.settings.supabase_bucket).download(remote_path)
        cache_path = self.settings.artifact_dir / Path(remote_path).name
        cache_path.write_bytes(data)
        return cache_path


def get_storage() -> ArtifactStorage:
    return ArtifactStorage()
