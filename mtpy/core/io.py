import gzip
import io
import joblib
import os
import pandas as pd
import re
import requests
import shutil
from urllib.parse import urlparse

from typing import Any, Literal, Optional

from .app import App

from .utils.helpers import (
    format_number,
    parse_size_bytes
)


class FileSystem(object):
    """
    FileSystem class to handle file read/write operations.

    Parameters
    ----------
    path : str
        The path to the file system root directory.
    """

    def __init__(self, path: str):
        self.path = path

    def is_url(self, name: str) -> bool:
        """
        Check if the given path is a valid URL.

        Parameters
        ----------
        name : str
            The name of the path to check.

        Returns
        -------
        bool
            Whether the path is a valid URL.
        """
        url = urlparse(name)

        return all([url.scheme, url.netloc])

    def exists(self, name: str) -> bool:
        """
        Check if the file exists in the file system.

        Parameters
        ----------
        name : str
            The name of the file to check.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        path = '{}/{}'.format(self.path, name)

        return os.path.isfile(path) or os.path.isdir(path)

    def remove(self, name: str) -> None:
        """
        Remove the given file from the file system.

        Parameters
        ----------
        name : str
            The name of the file to remove.
        """
        path = '{}/{}'.format(self.path, name)

        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    def listdir(self, name: Optional[str] = None) -> list[str]:
        """
        List the files and directories in a given directory.

        Parameters
        ----------
        name : str, optional
            The name of the directory to list.

        Returns
        -------
        list[str]
            The list of files and directories.
        """
        path = self.path
        if name is not None:
            path = '{}/{}'.format(path, name)

        return os.listdir(path)

    @staticmethod
    def assure_local_file(name: str) -> str:
        """
        Assure the existence of the given file in the local file system.

        Parameters
        ----------
        name : str
            The name of the file to check.

        Returns
        -------
        str
            The full absolute path to the file.
        """
        app = App.get_()
        fs_loc = FileSystem(app.fspath)

        if not fs_loc.exists(name):
            fs_rem = app.fs
            if fs_rem.exists(name):
                fs_loc.write_bytes(fs_rem.read_bytes(name), name)
            else:
                return

        return '{}/{}'.format(fs_loc.path, name)

    def read(
        self,
        name: str,
        format: Optional[Literal['csv', 'json', 'excel']] = None,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read the content of the given file.

        Parameters
        ----------
        name : str
            The name of the file to read.
        format : Literal['csv', 'json', 'excel'], optional
            The format of the file.
        compression : Literal['gzip'], optional
            The compression of the file.
        kwargs : dict
            Additional keyword arguments sent to the reader used.

        Returns
        -------
        pd.DataFrame
            The content of the file as a DataFrame.
        """
        if self.is_url(name):
            return self.read_url(
                name=name,
                format=format,
                compression=compression,
                **kwargs
            )

        path = '{}/{}'.format(self.path, name)
        encoding = kwargs.get('encoding')
        mode = 'rb' if compression == 'gzip' or format == 'excel' else 'r'

        with open(path, mode, encoding=encoding) as fh:
            if compression == 'gzip':
                with gzip.GzipFile(mode='rb', fileobj=fh) as gz:
                    buffer = io.StringIO(gz.read().decode('utf-8'))
            elif format == 'excel':
                buffer = io.BytesIO(fh.read())
            else:
                buffer = io.StringIO(fh.read())

        if format == 'csv':
            df = pd.read_csv(buffer, **kwargs)
        elif format == 'json':
            df = pd.read_json(buffer, **kwargs)
        elif format == 'excel':
            df = pd.read_excel(buffer, **kwargs)
        else:
            df = buffer.getvalue()

        return df

    def write(
        self,
        df: pd.DataFrame,
        name: str,
        format: Optional[Literal['csv', 'json', 'excel']] = None,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> str:
        """
        Write the content of the given DataFrame into a file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to write.
        name : str
            The name of the file to write.
        format : Literal['csv', 'json', 'excel'], optional
            The format of the file.
        compression : Literal['gzip'], optional
            The compression of the file.
        kwargs : dict
            Additional keyword arguments sent to the writer used.

        Returns
        -------
        str
            The relative path to the new file.
        """
        path = '{}/{}'.format(self.path, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if format == 'csv':
            buffer = io.StringIO()
            df.to_csv(buffer, **kwargs)
        elif format == 'json':
            buffer = io.StringIO()
            df.to_json(buffer, **kwargs)
        elif format == 'excel':
            buffer = io.BytesIO()
            df.to_excel(buffer, **kwargs)
        else:
            buffer = io.StringIO(df)

        if compression == 'gzip':
            gz_buffer = io.BytesIO()
            with gzip.GzipFile(mode='wb', fileobj=gz_buffer) as gz:
                if isinstance(buffer, io.StringIO):
                    gz.write(bytes(buffer.getvalue(), 'utf-8'))
                else:
                    gz.write(buffer.getvalue())
            buffer = gz_buffer

        mode = 'wb' if isinstance(buffer, io.BytesIO) else 'w'

        with open(path, mode) as fh:
            fh.write(buffer.getvalue())

        return path

    def read_bytes(self, name: str) -> bytes:
        """
        Read the content of the given file as bytes.

        Parameters
        ----------
        name : str
            The name of the file to read.

        Returns
        -------
        bytes
            The content of the file as bytes.
        """
        path = '{}/{}'.format(self.path, name)

        with open(path, 'rb') as fh:
            content = fh.read()

        return content

    def write_bytes(self, content: bytes, name: str) -> str:
        """
        Write the given content to a file as bytes.

        Parameters
        ----------
        content : bytes
            The content to write.
        name : str
            The name of the file to write.

        Returns
        -------
        str
            The relative path to the new file.
        """
        path = '{}/{}'.format(self.path, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as fh:
            fh.write(content)

        return path

    def save_object(self, value: Any, name: str) -> str:
        """
        Save the given object to a file as bytes using joblib library.

        Parameters
        ----------
        value : Any
            The object to save.
        name : str
            The name of the file to save.

        Returns
        -------
        str
            The relative path to the new file.
        """
        path = '{}/{}'.format(self.path, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump(value, path)

        return path

    def load_object(self, name: str) -> Any:
        """
        Load object from a file as bytes using joblib library.

        Parameters
        ----------
        name : str
            The name of the file to load.

        Returns
        -------
        Any
            The loaded object.
        """
        path = '{}/{}'.format(self.path, name)

        return joblib.load(path)

    def read_url(
        self,
        name: str,
        format: Optional[Literal['csv', 'json', 'excel']] = None,
        compression: Optional[Literal['gzip']] = None,
        chunk_size: Optional[int] = None,
        max_retries: int = 0,
        errors: Literal['raise', 'warn', 'ignore'] = 'raise',
        verbose: int = 0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read the content of the given URL. In case of a large file,
        tries to download the file in chunks.

        Parameters
        ----------
        name : str
            The URL to read.
        format : Literal['csv', 'json', 'excel'], optional
            The format of the file.
        compression : Literal['gzip'], optional
            The compression of the file.
        chunk_size : int, optional
            The chunk size to use.
        max_retries : int, default 0
            The maximum number of retries.
        errors : Literal['raise', 'warn', 'ignore'], default 'raise'
            The error handling strategy.
        verbose : int, default 0
            The verbosity level.
        kwargs : dict
            Additional keyword arguments sent to the reader used.

        Returns
        -------
        pd.DataFrame
            The content of the URL as a DataFrame.
        """
        if chunk_size is None:
            chunk_size = 0
        elif isinstance(chunk_size, str):
            chunk_size = parse_size_bytes(chunk_size)

        if verbose > 0:
            print('Get HTTP headers...')

        with requests.head(name, headers={'Accept-Encoding': None}, verify=False) as r:
            r.raise_for_status()
            url_headers = r.headers

        use_ranges = 'bytes' in url_headers.get('Accept-Ranges', '').split(',')
        file_size = int(url_headers.get('Content-Length') or 0)

        if file_size == 0:
            raise IOError('Unable to determine file size')

        if verbose > 0:
            print('    File Size: {} bytes | Use Ranges: {} | Chunk Size: {} bytes'.format(
                format_number(file_size),
                use_ranges,
                format_number(chunk_size)
            ))
            print('Read URL...')

        buffer = io.BytesIO()
        buffer_size = 0

        if use_ranges:
            while buffer_size < file_size:
                stream = chunk_size > 0
                headers = {'Accept-Encoding': None}
                if buffer_size > 0:
                    headers['Range'] = 'bytes={:d}-'.format(buffer_size)

                with requests.get(name, headers=headers, stream=stream, verify=False) as r:
                    r.raise_for_status()
                    if stream:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            buffer.write(chunk)
                            buffer_size = len(buffer.getbuffer())
                            if verbose > 0:
                                print('    Chunk: {} bytes ({}%)'.format(
                                    format_number(buffer_size),
                                    format_number(100 * buffer_size / file_size)
                                ))
                    else:
                        buffer.write(r.content)
                        buffer_size = len(buffer.getbuffer())
                        if verbose > 0:
                            print('    Range: {} bytes ({}%)'.format(
                                format_number(buffer_size),
                                format_number(100 * buffer_size / file_size)
                            ))
        else:
            n_req = 0
            while buffer_size < file_size and n_req <= max_retries:
                with requests.get(name, verify=False) as r:
                    r.raise_for_status()
                    if len(r.content) > buffer_size:
                        buffer = io.BytesIO(r.content)
                    buffer_size = len(buffer.getbuffer())
                    if verbose > 0:
                        print('    Content: {} bytes ({}%)'.format(
                            format_number(buffer_size),
                            format_number(100 * buffer_size / file_size)
                        ))

                n_req += 1

        buffer.seek(0)
        if buffer_size < file_size:
            msg = 'IncompleteRead: {} bytes ({}%)'.format(
                format_number(buffer_size),
                format_number(100 * buffer_size / file_size)
            )
            if errors == 'raise':
                raise IOError(msg)
            elif errors == 'warn' or verbose > 0:
                print(msg)

        if verbose > 0:
            print('Process data...')

        if format == 'csv':
            if verbose > 0:
                print('    Format: CSV')
            df = pd.read_csv(buffer, engine='python', on_bad_lines='skip', **kwargs)
            if buffer_size < file_size:
                df = df.iloc[:-1]
        elif format == 'json':
            if verbose > 0:
                print('    Format: JSON')
            df = pd.read_json(buffer, **kwargs)
        else:
            m = re.match(r'.*charset\=([A-Za-z0-9\-]+)', url_headers.get('Content-Type'))
            encoding = m.group(1) if m is not None else 'utf-8'
            if verbose > 0:
                print('    Format: {}'.format(encoding.upper()))
            df = str(buffer.getbuffer(), encoding)

        if verbose > 0:
            print('Complete!')

        return df

    def read_csv(
        self,
        name: str,
        sep: str = ',',
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read the content of the given CSV file.

        Parameters
        ----------
        name : str
            The name of the file to read.
        sep : str, default ','
            String of length 1. Field delimiter for the output file.
        compression : Literal['gzip'], optional
            Compression type for the file.
        kwargs : dict
            Additional keyword arguments sent to the reader used.

        Returns
        -------
        pd.DataFrame
            The content of the file as a DataFrame.
        """
        return self.read(
            name,
            format='csv',
            compression=compression,
            sep=sep,
            **kwargs
        )

    def write_csv(
        self,
        df: pd.DataFrame,
        name: str,
        sep: str = ',',
        index: bool = False,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> str:
        """
        Write the content of the given DataFrame into a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to write.
        name : str
            The name of the file to write.
        sep : str, default ','
            String of length 1. Field delimiter for the output file.
        index : bool, default False
            Write row names (index).
            See the pandas.to_csv method for more information.
        compression : Literal['gzip'], optional
            Compression type for the file.
        kwargs : dict
            Additional keyword arguments sent to the writer used.

        Returns
        -------
        str
            The relative path to the new file.
        """
        return self.write(
            df,
            name,
            format='csv',
            compression=compression,
            sep=sep,
            index=index,
            **kwargs
        )

    def read_json(
        self,
        name: str,
        orient: Literal['records', 'index', 'columns', 'split', 'values', 'table'] = 'records',
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read the content of the given JSON file.

        Parameters
        ----------
        name : str
            The name of the file to read.
        orient : Literal['records', 'index', 'columns', 'split', 'values', 'table'], default 'records'
            Indication of expected JSON string format.
            See the pandas.read_json method for more information.
        compression : Literal['gzip'], optional
            Compression type for the file.
        kwargs : dict
            Additional keyword arguments sent to the reader used.

        Returns
        -------
        pd.DataFrame
            The content of the file as a DataFrame.
        """
        return self.read(
            name,
            format='json',
            compression=compression,
            orient=orient,
            **kwargs
        )

    def write_json(
        self,
        df: pd.DataFrame,
        name: str,
        orient: Literal['records', 'index', 'columns', 'split', 'values', 'table'] = 'records',
        date_format: Literal['iso', 'epoch'] = 'iso',
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> str:
        """
        Write the content of the given DataFrame into a JSON file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to write.
        name : str
            The name of the file to write.
        orient : Literal['records', 'index', 'columns', 'split', 'values', 'table'], default 'records'
            Indication of expected JSON string format.
        date_format : Literal['iso', 'epoch'], default 'iso'
            The date format to use.
        compression : Literal['gzip'], optional
            Compression type for the file.
        kwargs : dict
            Additional keyword arguments sent to the writer used.

        Returns
        -------
        str
            The relative path to the new file.
        """
        return self.write(
            df,
            name,
            format='json',
            compression=compression,
            orient=orient,
            date_format=date_format,
            **kwargs
        )

    def read_excel(
        self,
        name: str,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read the content of the given Excel file.

        Parameters
        ----------
        name : str
            The name of the file to read.
        compression : Literal['gzip'], optional
            Compression type for the file.
        kwargs : dict
            Additional keyword arguments sent to the reader used.

        Returns
        -------
        pd.DataFrame
            The content of the file as a DataFrame.
        """
        return self.read(
            name,
            format='excel',
            compression=compression,
            **kwargs
        )

    def write_excel(
        self,
        df: pd.DataFrame,
        name: str,
        index: bool = False,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> str:
        """
        Write the content of the given DataFrame into an Excel file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to write.
        name : str
            The name of the file to write.
        index : bool, default False
            Whether to write row indices into the file.
        compression : Literal['gzip'], optional
            Compression type for the file.
        kwargs : dict
            Additional keyword arguments sent to the writer used.

        Returns
        -------
        str
            The relative path to the new file.
        """
        return self.write(
            df,
            name,
            format='excel',
            compression=compression,
            index=index,
            **kwargs
        )
