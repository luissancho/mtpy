import gzip
import io
import joblib
import pandas as pd
from s3fs import S3FileSystem
import tempfile

from typing import Any, Literal, Optional
from typing_extensions import Self

from ..io import FileSystem


class S3(FileSystem):

    def __init__(self, aws_params, bucket):
        super().__init__(path='/')

        self._client = None

        self.aws_params = aws_params
        self.bucket = bucket

    def is_connected(self) -> bool:
        return self._client is not None

    def connect(self, force: bool = False) -> Self:
        if self.is_connected():
            if force:
                self.disconnect()
            else:
                return self
        
        self.connection()
        
        return self

    def disconnect(self) -> Self:
        if self.is_connected():
            self._client = None

        return self

    def connection(self) -> Self:
        self._client = S3FileSystem(
            key=self.aws_params['key'],
            secret=self.aws_params['secret']
        )

    def exists(self, name: str) -> bool:
        self.connect()

        path = '{}/{}'.format(self.bucket, name)

        return self._client.exists(path)

    def remove(self, name: str) -> None:
        self.connect()

        path = '{}/{}'.format(self.bucket, name)

        self._client.rm(path, recursive=True)

    def listdir(self, name: Optional[str] = None) -> list[str]:
        self.connect()

        path = self.bucket
        if name is not None:
            path = '{}/{}'.format(path, name)

        return [
            i.replace(path + '/', '')
            for i in self._client.ls(path)
            if i.rstrip('/') != path
        ]

    def read(
        self,
        name: str,
        format: Optional[Literal['csv', 'json', 'excel']] = None,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> pd.DataFrame:
        if self.is_url(name):
            return self.read_url(name, format=format, compression=compression, **kwargs)

        self.connect()

        path = '{}/{}'.format(self.bucket, name)
        content = io.StringIO()

        mode = 'r'
        if compression == 'gzip':
            mode = 'rb'

        with self._client.open(path, mode) as fh:
            if compression == 'gzip':
                with gzip.GzipFile(mode='rb', fileobj=fh) as gz:
                    content = io.StringIO(gz.read().decode('UTF-8'))
            else:
                content = io.StringIO(fh.read())

            if format == 'csv':
                df = pd.read_csv(content, **kwargs)
            elif format == 'json':
                df = pd.read_json(content, **kwargs)
            else:
                df = content.getvalue()

        return df

    def write(
        self,
        df: pd.DataFrame,
        name: str,
        format: Optional[Literal['csv', 'json', 'excel']] = None,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> str:
        self.connect()

        path = '{}/{}'.format(self.bucket, name)
        content = io.StringIO()

        if format == 'csv':
            df.to_csv(content, **kwargs)
        elif format == 'json':
            df.to_json(content, **kwargs)
        else:
            content = io.StringIO(df)

        mode = 'w'
        if compression == 'gzip':
            mode = 'wb'
            gz_content = io.BytesIO()
            with gzip.GzipFile(mode='wb', fileobj=gz_content) as gz:
                gz.write(bytes(content.getvalue(), 'utf-8'))
            content = gz_content

        with self._client.open(path, mode) as fh:
            fh.write(content.getvalue())

        return path

    def read_bytes(self, name: str) -> bytes:
        self.connect()

        path = '{}/{}'.format(self.bucket, name)

        with self._client.open(path, 'rb') as fh:
            content = fh.read()

        return content

    def write_bytes(self, content: bytes, name: str) -> str:
        self.connect()

        path = '{}/{}'.format(self.bucket, name)

        with self._client.open(path, 'wb') as fh:
            fh.write(content)

        return path

    def save_object(self, value: Any, name: str) -> str:
        with tempfile.TemporaryFile() as tf:
            joblib.dump(value, tf)
            tf.seek(0)
            path = self.write_bytes(tf.read(), name)

        return path

    def load_object(self, name: str) -> Any:
        with tempfile.TemporaryFile() as tf:
            tf.write(self.read_bytes(name))
            tf.seek(0)
            value = joblib.load(tf)

        return value
