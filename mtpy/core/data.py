from dask import dataframe as dd
from importlib import import_module
import json
import numpy as np
import pandas as pd
import requests
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, ISOLATION_LEVEL_DEFAULT
from sqlalchemy import create_engine, exc, text
from sshtunnel import SSHTunnelForwarder

from typing import Any, Literal, Optional
from typing_extensions import Self

from .app import Core

from .utils.dates import is_datetime
from .utils.helpers import (
    format_data,
    is_empty,
    is_json,
    is_number,
    is_string,
    parse_meta
)


class Adapter(Core):
    """
    Abstract class for generic data service adapters.
    
    Parameters
    ----------
    params : dict, optional
        Service connection parameters.
    """

    def __init__(self, params: Optional[dict] = None):
        if type(self) is Adapter:
            raise TypeError('Abstract class should not be instantiated.')

        super().__init__()

        self.params = params
        self._client = None

    def _build_client(self) -> Self:
        """
        Here goes the specific creation logic of the service client.
        """
        return self

    def is_built(self) -> bool:
        """
        Check if the client has been created and exists.
        """
        return not is_empty(self._client)
    
    def build(self, force: bool = False) -> Self:
        """
        Build service client.

        Parameters
        ----------
        force : bool, optional
            Whether to force the creation even if it is already created.
        """
        if self.is_built() and not force:
            return self
        
        self._build_client()
        
        return self

    def dispose(self) -> Self:
        """
        Remove the service client object.
        """
        if self.is_built():
            self._client = None

        return self


class APIAdapter(Adapter):
    """
    Abstract class for API service adapters,
    which will have Requests library as handler for endpoint API calls.
    
    Parameters
    ----------
    params : dict, optional
        Service connection parameters.
    
    Attributes
    ----------
    api_url : str
        Base URL for the API service.
    """

    api_url = None

    def __init__(self, params: Optional[dict] = None):
        if type(self) is APIAdapter:
            raise TypeError('Abstract class should not be instantiated.')

        super().__init__(params=params)
    
    def execute(
        self,
        method: str = 'GET',
        endpoint: str = '',
        params: Optional[dict] = None,
        data: Optional[str | dict] = None,
        headers: Optional[dict] = None,
        **kwargs
    ) -> Any:
        """
        Execute a request to the API.

        Parameters
        ----------
        method : str, default 'GET'
            Method to be passed to the execution client.
        endpoint : str
            API endpoint to be passed to the execution client.
        params : dict, optional
            Parameters to be passed to the execution client.
        data : str | dict, optional
            Data to be passed to the execution client.
            If a string is passed, it is assumed to be a JSON string.
            If a dictionary is passed, it will be converted to a JSON object.
        headers : dict, optional
            Headers to be passed to the execution client.
        kwargs : dict
            Additional arguments to be passed to the execution client.
        
        Returns
        -------
        Any
            Result of the request execution.
        """
        method = method.lower()
        url = f'{self.api_url}/{endpoint}'

        r = requests.request(
            method=method,
            url=url,
            params=params,
            data=data,
            headers=headers,
            **kwargs
        )

        if not r.status_code == requests.codes.ok:
            raise r.raise_for_status()

        return r.json()


class DBAdapter(Adapter):
    """
    Abstract class for database adapters.
    
    Parameters
    ----------
    params : dict, optional
        Adapter connection parameters.
    database : str, optional
        Adapter name.
    """

    def __init__(self, params: Optional[dict] = None, database: Optional[str] = None):
        if type(self) is DBAdapter:
            raise TypeError('Abstract class should not be instantiated.')

        super().__init__(params=params)

        self.database = database
    
    def _execute_client(self, **kwargs) -> Any:
        """
        Here goes the specific execution logic of the service client.

        Parameters
        ----------
        kwargs : dict
            Request arguments.
        
        Returns
        -------
        Any
            Result of the request execution.
        """
        return

    def execute(
        self,
        **kwargs
    ) -> Any:
        """
        Execute a query in the database.

        Parameters
        ----------
        kwargs : dict
            Request arguments to be passed to the execution handler.
        
        Returns
        -------
        Any
            Result of the request execution.
        """
        self.connect()

        try:
            result = self._execute_client(**kwargs)
        except Exception as e:
            raise e
        finally:
            self.disconnect()

        return result
    
    def build_select_query(
        self,
        table: str,
        columns: Optional[str | list[str]] = None,
        relations: Optional[list[dict]] = None,
        filters: Optional[list[str] | dict] = None,
        sort: Optional[str | list[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> str | dict:
        return
    
    def build_agg_query(
        self,
        table: str,
        columns: Optional[str | list[str] | dict[str, str]] = None,
        agg: Optional[str] = 'count',
        groupby: Optional[str | list[str] | dict[str, str]] = None,
        filters: Optional[list[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> str:
        return

    def get_results(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> pd.DataFrame:
        return
    
    def get_row(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> dict:
        return
    
    def get_var(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> Any:
        return
    
    def get_agg(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> Any:
        return
    
    def get_count(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> Any:
        return
    
    def get_tables(
        self,
        schema: Optional[str] = None
    ) -> list[str]:
        return
    
    def get_columns(
        self,
        table: str
    ) -> list[str]:
        return
    
    def get_meta(
        self,
        table: str
    ) -> dict[str, dict]:
        return

    def truncate(
        self,
        table: str
    ) -> Self:
        return self

    def drop(
        self,
        table: str
    ) -> Self:
        return self

class SQLAdapter(DBAdapter):
    """
    Abstract class for SQL database adapters,
    which will have a SQLAlchemy engine as handler for database operations.
    
    Parameters
    ----------
    params : dict
        Database connection parameters.
    database : str
        Database name.

    Attributes
    ----------
    driver : str
        Database driver prefix for SQLAlchemy.
    default_port : int
        Default database port.
    quote : str
        Quote character used for wrapping column and table names in SQL queries.
    sessions : bool
        Whether to use sessions for transactions.
    data_types : dict
        Mapping of data types between Python and SQL.
    bool_as_int : bool
        Whether to use integers for boolean values.
    autokey_suffix : str
        Auto-increment key suffix string CREATE TABLE.
    date_groupings : dict
        Mapping of date groupings for aggregation queries.
    on_conflict_suffix : str
        Suffix for ON CONFLICT clause in UPSERT queries.
    on_conflict_update : str
        Update format for ON CONFLICT clause in UPSERT queries.
    create_relations : str
        Relation clause format for CREATE TABLE.
    create_suffix : str
        Suffix for CREATE TABLE.
    """

    driver = None
    default_port = None
    quote = '"'
    sessions = True
    data_types = {
        'varchar': 'VARCHAR',
        'text': 'TEXT',
        'date': 'DATE',
        'datetime': 'DATETIME',
        'numeric': 'NUMERIC',
        'tinyint': 'TINYINT',
        'smallint': 'SMALLINT',
        'mediumint': 'MEDIUMINT',
        'integer': 'INTEGER',
        'bigint': 'BIGINT',
        'boolean': 'BOOLEAN'
    }
    bool_as_int = False
    autokey_suffix = None
    date_groupings = {
        'year': "DATE_TRUNC('YEAR', {col})",
        'month': "DATE_TRUNC('MONTH', {col})",
        'week': "DATE_TRUNC('WEEK', {col})",
        'day': "DATE_TRUNC('DAY', {col})"
    }
    on_conflict_suffix = 'ON CONFLICT ({key}) DO UPDATE SET'
    on_conflict_update = '{col} = EXCLUDED.{col}'
    create_relations = 'FOREIGN KEY ({key}) REFERENCES {table}'
    create_suffix = None

    def __init__(self, params: dict, database: str):
        if type(self) is SQLAdapter:
            raise TypeError('Abstract class should not be instantiated.')

        super().__init__(params=params, database=database)

        self._connection = None
        self._session = None
        self._ssh = None

        # Stage directory path, where data files are stored before loading and unloading data.
        self.stage_dir = self.params.get('stage_dir')
    
    @property
    def stage(self) -> bool:
        """
        Whether to use a stage directory for loading and unloading data.
        """
        return self.stage_dir is not None

    def _build_client(self) -> Self:
        db_params = dict(
            host=self.params.get('host'),
            port=int(self.params.get('port', self.default_port)),
            username=self.params.get('username'),
            password=self.params.get('password'),
            database=self.database
        )

        if 'ssh' in self.params and self.params['ssh'].get('host'):
            if 'local_bind' in self.params['ssh'] and self.params['ssh']['local_bind'].get('host'):
                local_bind_address = (
                    self.params['ssh']['local_bind']['host'],
                    int(self.params['ssh']['local_bind'].get('port', 22))
                )
            else:
                local_bind_address = None
            
            self._ssh = SSHTunnelForwarder(
                (
                    self.params['ssh'].get('host'),
                    int(self.params['ssh'].get('port', 22))
                ),
                ssh_username=self.params['ssh'].get('username'),
                ssh_private_key=self.params['ssh'].get('pkey'),
                ssh_private_key_password=self.params['ssh'].get('passphrase'),
                local_bind_address=local_bind_address,
                remote_bind_address=(
                    db_params['host'],
                    db_params['port']
                )
            )
            self._ssh.start()

            db_params |= dict(
                host=self._ssh.local_bind_host,
                port=self._ssh.local_bind_port
            )

        self._client = create_engine(
            '{}://{username}:{password}@{host}:{port}/{database}'.format(self.driver, **db_params),
            pool_pre_ping=True
        )

        return self
    
    def _connect_client(self) -> Self:
        """
        Connect to the database handler.
        """
        self._connection = self._client.connect()

        return self

    def _execute_client(
        self,
        query: str,
        autocommit: bool = False
    ) -> Any:
        """
        Execute a query in the database handler.

        Parameters
        ----------
        query : str
            SQL query to execute.
        autocommit : bool
            Whether to execute the query in autocommit mode.
        
        Returns
        -------
        Any
            Result of the query execution.
        """
        query = self.parse_query(query)
        
        if autocommit and self.sessions:
            connection = self._client.raw_connection()
            cursor = connection.cursor()

            try:
                connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

                result = cursor.execute(query)

                connection.set_isolation_level(ISOLATION_LEVEL_DEFAULT)
            except Exception as e:
                raise e
            finally:
                cursor.close()
                connection.close()
        else:
            result = self._connection.execute(text(query))
        
        return result

    def is_connected(self) -> bool:
        """
        Check if the adapter is connected to the database.
        """
        return self._connection is not None

    def in_session(self) -> bool:
        """
        Check if the adapter is in a transaction session.
        """
        return self._session is not None
    
    def in_ssh(self) -> bool:
        """
        Check if the adapter is using an SSH tunnel.
        """
        return self._ssh is not None and self._ssh.is_active

    def connect(self, force: bool = False) -> Self:
        """
        Connect to the database.
        """
        if self.is_connected():
            if force:
                self.disconnect()
            else:
                return self
        
        self._build_client()._connect_client()
        
        return self

    def disconnect(self) -> Self:
        """
        Disconnect from the database.
        """
        if self.in_session():
            self._session.close()
            self._session = None

        if self.is_connected():
            if hasattr(self._connection, 'close'):
                self._connection.close()
            self._connection = None

        if self.in_ssh():
            self._ssh.stop()
            self._ssh = None

        return self

    def begin(self) -> Self:
        """
        Begin a transaction session.
        """
        if not self.sessions:
            return self

        self.connect()

        self._session = self._client.begin()

        return self

    def commit(self) -> Self:
        """
        Commit the current transaction session.
        """
        if not self.sessions:
            return self

        if self.in_session():
            self._session.commit()

        self.disconnect()

        return self

    def rollback(self) -> Self:
        """
        Rollback the current transaction session.
        """
        if not self.sessions:
            return self

        if self.in_session():
            self._session.rollback()

        self.disconnect()

        return self

    def execute(
        self,
        query: str,
        **kwargs
    ) -> Any:
        """
        Execute a query in the database.

        Parameters
        ----------
        query : str
            SQL query to execute.
        kwargs : dict
            Additional arguments to be passed to the execution handler.
        
        Returns
        -------
        Any
            Result of the query execution.
        """
        self.connect()

        try:
            result = self._execute_client(query=query, **kwargs)
        except exc.DatabaseError as e:
            raise e.orig
        except Exception as e:
            raise e
        finally:
            if not self.in_session():
                self.disconnect()

        return result
    
    def execute_load(
        self,
        query: str,
        path: Optional[str] = None
    ) -> Self:
        """
        Execute a load query in the database.

        Parameters
        ----------
        query : str
            SQL load/copyto query to execute.
        path : str, optional
            Path to the file to use in the load query (if applicable).
        """
        return self.execute(query, autocommit=True)

    def execute_unload(
        self,
        query: str,
        path: Optional[str] = None
    ) -> Self:
        """
        Execute an unload query in the database.

        Parameters
        ----------
        query : str
            SQL unload/copyfrom query to execute.
        path : str, optional
            Path to the file to use in the unload query (if applicable).
        """
        return self.execute(query, autocommit=True)
    
    def table_exists(self, table: str) -> bool:
        """
        Check if a table exists in the database.

        Parameters
        ----------
        table : str
            Table name to check.
        
        Returns
        -------
        bool
            Whether the table exists.
        """
        self.connect()

        result = self._client.has_table(table)

        self.disconnect()

        return result
    
    def parse_query(self, query: str) -> str:
        """
        Parse a query string in order to replace quote chars and make it compatible with the database.

        Parameters
        ----------
        query : str
            Query string to parse.
        
        Returns
        -------
        str
            Parsed query string.
        """
        if self.quote != '"':
            query = query.replace('"', self.quote)

        return query

    def table_stage_exists(self, table: str) -> bool:
        """
        Check if a table stage directory exists.

        Parameters
        ----------
        table : str
            Table name to check.
        
        Returns
        -------
        bool
            Whether the table stage directory exists.
        """
        if not self.stage:
            return False
        
        path = '{}/{}'.format(self.stage_dir, table)

        return self.app.fs.exists(path)

    def table_stage_clean(self, table: str) -> Self:
        """
        Clean a table stage directory.

        Parameters
        ----------
        table : str
            Table name whose stage directory must be cleaned.
        """
        if not self.stage:
            return self

        path = '{}/{}'.format(self.stage_dir, table)

        if self.app.fs.exists(path):
            self.app.fs.remove(path)

        return self

    def get_table_parts(self, table: str) -> tuple[str, str]:
        """
        Split table name into schema and name components.

        Parameters
        ----------
        table : str
            Table name to split.
        
        Returns
        -------
        tuple[str, str]
            Schema and name parts of the table name.
        """
        if '.' in table:
            schema, name = table.split('.', 1)
        else:
            schema = None
            name = table

        return schema, name

    def get_table_name(self, table: str) -> str:
        """
        Get only the name of the table excluding its schema component.

        Parameters
        ----------
        table : str
            Table name.
        
        Returns
        -------
        str
            Name component of the table name.
        """
        _, name = self.get_table_parts(table)

        return name

    def build_create_query(
        self,
        table: str,
        meta: dict,
        key: Optional[str | list[str]] = None,
        relations: Optional[dict] = None,
        indexes: Optional[dict] = None,
        autokey: bool = False,
        sortkey: Optional[str | list[str]] = None,
        diststyle: Optional[str | list[str]] = None,
        distkey: Optional[str | list[str]] = None,
        temp: bool = False
    ) -> str:
        if key is None:
            key = [list(meta)[0]]
        elif not isinstance(key, (list, tuple)):
            key = [key]

        query = ''

        query += 'CREATE'
        if temp:
            query += ' TEMPORARY'
        query += ' TABLE IF NOT EXISTS {} (\n'.format(table)

        lines = []

        for name, desc in meta.items():
            dtyp = desc.get('type')
            dlen = desc.get('length')
            dnul = desc.get('nullable', True)
            ddef = desc.get('default')

            line = '"{}"'.format(name)

            if dtyp in ['str', 'cat', 'obj']:
                if dlen == 0:
                    line += ' {}'.format(self.data_types['text'])
                else:
                    line += ' {}({})'.format(self.data_types['varchar'], dlen)
            elif dtyp == 'dtd':
                line += ' {}'.format(self.data_types['date'])
            elif dtyp == 'dts':
                line += ' {}'.format(self.data_types['datetime'])
            elif dtyp == 'num':
                line += ' {}({}, {})'.format(self.data_types['numeric'], *dlen)
            elif dtyp == 'int':
                if dlen == 1:
                    line += ' {}'.format(self.data_types['tinyint'])
                elif dlen == 2:
                    line += ' {}'.format(self.data_types['smallint'])
                elif dlen == 3:
                    line += ' {}'.format(self.data_types['mediumint'])
                elif dlen == 4:
                    line += ' {}'.format(self.data_types['integer'])
                else:
                    line += ' {}'.format(self.data_types['bigint'])
            elif dtyp == 'bin':
                line += ' {}'.format(self.data_types['boolean'])
                if ddef is not None:
                    if self.bool_as_int:
                        ddef = 1 if ddef else 0
                    else:
                        ddef = 'TRUE' if ddef else 'FALSE'
            else:
                continue

            if dnul is False or name in key:
                line += ' NOT NULL'
                if autokey and name == key[0] and self.autokey_suffix is not None:
                    line += ' {}'.format(self.autokey_suffix)
                elif ddef is not None:
                    line += ' DEFAULT {}'.format(ddef)
            else:
                line += ' NULL'

            lines.append(line)

        lines.append('PRIMARY KEY ("{}")'.format('", "'.join(key)))

        if isinstance(indexes, str):
            indexes = {indexes: indexes}
        elif isinstance(indexes, (list, tuple)):
            indexes = {idx: idx for idx in indexes}

        if isinstance(indexes, dict) and len(indexes) > 0:
            for name, idx in indexes.items():
                if isinstance(idx, str):
                    idx = idx.split(',')

                lines.append('KEY "{}" ("{}")'.format(name, '", "'.join(idx)))

        if isinstance(relations, dict) and len(relations) > 0:
            for _, desc in relations.items():
                name = '"{}_{}_fk"'.format(**desc) if '{name}' in self.create_relations else ''
                line = self.create_relations.format(name=name, **desc)
                if desc.get('column') is not None:
                    line += ' ("{}")'.format(desc['column'])

                lines.append(line)

        query += ',\n'.join(['    {}'.format(ln) for ln in lines])

        query += '\n)'
        if self.create_suffix is not None:
            suffix = self.create_suffix
            if '{sortkey}' in suffix and sortkey is not None:
                if isinstance(sortkey, str):
                    sortkey = sortkey.split(',')
                suffix = suffix.replace('{sortkey}', '"{}"'.format('", "'.join(sortkey)))
            if '{diststyle}' in suffix and diststyle is not None:
                suffix = suffix.replace('{diststyle}', '"{}"'.format(diststyle))
            if '{distkey}' in suffix and distkey is not None:
                suffix = suffix.replace('{distkey}', '"{}"'.format(distkey))

            query += ' {}'.format(suffix)
        
        return query

    def build_clone_query(
        self,
        table: str,
        name: str,
        temp: bool = False
    ) -> str:
        query = ''

        query += 'CREATE'
        if temp:
            query += ' TEMPORARY'
        query += ' TABLE IF NOT EXISTS {} (LIKE {} INCLUDING ALL)'.format(name, table)

        return query

    def build_load_query(
        self,
        table: str,
        header: bool = False,
        fmt: Optional[str] = 'csv',
        sep: Optional[str] = ',',
        stage: Optional[str] = None,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> tuple[str, str]:
        return None, None

    def build_unload_query(
        self,
        table: str,
        query: str,
        header: bool = False,
        fmt: Optional[str] = 'csv',
        sep: Optional[str] = ',',
        stage: Optional[str] = None,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> tuple[str, str]:
        return None, None

    def build_insert_query(
        self,
        table: str,
        data: pd.DataFrame
    ) -> str:
        query = 'INSERT INTO {} ("{}")\nVALUES\n'.format(table, '", "'.join(data.columns))

        value_map = {i: 'NULL' for i in ['None', '<NA>', 'nan', 'NaT']}
        if self.bool_as_int:
            value_map |= {'True': '1', 'False': '0'}

        rows = []
        for _, r in data.iterrows():
            row = r.astype(str, errors='ignore').replace(value_map).replace(r"'", "\\'", regex=True).replace(r':', r'\:', regex=True).values
            rows.append("    ('{}')".format("', '".join(row)).replace("'NULL'", 'NULL').replace("'DEFAULT'", 'DEFAULT'))

        query += '{}'.format(',\n'.join(rows))
        
        return query

    def build_upsert_query(
        self,
        table: str,
        data: pd.DataFrame,
        key: Optional[str | list[str]] = None
    ) -> str:
        if key is None:
            key = [list(data.columns)[0]]
        elif not isinstance(key, (list, tuple)):
            key = [key]

        query = self.build_insert_query(table, data) + '\n'

        if '{key}' in self.on_conflict_suffix:
            query += '{}\n'.format(
                self.on_conflict_suffix.format(key=', '.join(['"{}"'.format(col) for col in key]))
            )
        else:
            query += '{}\n'.format(self.on_conflict_suffix)

        query += ',\n'.join(['    {}'.format(
            self.on_conflict_update.format(col='"{}"'.format(col))
        ) for col in data.columns if col not in key])
        
        return query
    
    def build_select_query(
        self,
        table: str,
        columns: Optional[str | list[str]] = None,
        relations: Optional[list[dict]] = None,
        filters: Optional[list[str] | dict] = None,
        sort: Optional[str | list[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> str | dict:
        if isinstance(columns, str):
            columns = columns.split(',')

        if isinstance(columns, (list, tuple)) and len(columns) > 0:
            columns = [
                '"{}" AS "{}"'.format('"."'.join(col), '__'.join(col)) if isinstance(col, (list, tuple))
                else '"{}" AS "{}"'.format('"."'.join(col.split('__')), col) if '__' in col
                else col if '"' in col or self.quote in col
                else '"t"."{}"'.format(col)
                for col in columns
            ]
            select = '    {}'.format(',\n    '.join(columns))
        else:
            select = '    "t".*'

        query = 'SELECT\n{}\nFROM {} AS "t"'.format(select, table)

        if isinstance(relations, (list, tuple)) and len(relations) > 0:
            for rel in relations:
                if 'key' not in rel:
                    rel['key'] = rel['model'].key
                query += '\nLEFT JOIN {} AS "{}" ON '.format(rel['model'].table, rel['name'])
                if isinstance(rel['key'], (list, tuple)):
                    query += ' AND '.join([
                        '"{}"."{}" = "t"."{}"'.format(rel['name'], rel['model'].key[i], rel['key'][i])
                        for i in np.arange(len(rel['key']))
                    ])
                else:
                    query += '"{}"."{}" = "t"."{}"'.format(rel['name'], rel['model'].key, rel['key'])

        if isinstance(filters, (list, tuple)) and len(filters) > 0:
            query += '\nWHERE {}'.format('\n    AND '.join(filters))

        if isinstance(sort, str):
            sort = sort.split(',')

        if isinstance(sort, (list, tuple)) and len(sort) > 0:
            sort = [
                '"{}" {}'.format(*s.split(' ')) if ' ' in s
                else '"{}"'.format(s)
                for s in sort
            ]
            query += '\nORDER BY {}'.format(', '.join(sort))

        if limit:
            query += '\nLIMIT {:d}'.format(limit)

        if offset:
            query += '\nOFFSET {:d}'.format(offset)
        
        return query
    
    def build_agg_query(
        self,
        table: str,
        columns: Optional[str | list[str] | dict[str, str]] = None,
        agg: Optional[str] = 'count',
        groupby: Optional[str | list[str] | dict[str, str]] = None,
        filters: Optional[list[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> str:
        columns_ = {}
        if isinstance(columns, str):
            columns = columns.split(',')

        if isinstance(columns, dict) and len(columns) > 0:
            columns_ = {
                '"{}__{}"'.format(key, val.lower()): key if '"' in key or self.quote in key
                else '{}("t"."{}")'.format(val.upper(), key)
                for key, val in columns.items()
            }
        elif isinstance(columns, (list, tuple)) and len(columns) > 0:
            columns_ = {
                '"{}__{}"'.format(key, agg.lower()): key if '"' in key or self.quote in key
                else '{}("t"."{}")'.format(agg.upper(), key)
                for key in columns
            }
        elif isinstance(columns, str) and len(columns) > 0:
            columns_ = {
                '"{}__{}"'.format(columns, agg.lower()): columns if '"' in columns or self.quote in columns
                else '{}("t"."{}")'.format(agg.upper(), columns)
            }
        else:
            columns_ = {
                agg.lower(): '{}(*)'.format(agg.upper())
            }

        groupby_ = {}
        if isinstance(groupby, dict) and len(groupby) > 0:
            for key, val in groupby.items():
                name = key.replace('"', '').replace(self.quote, '')
                col = key if '"' in key or self.quote in key else '"t"."{}"'.format(key)
                val = val.lower()
                if val in ['year', 'month', 'week', 'day']:
                    groupby_['"{}__{}"'.format(name, val)] = self.date_groupings[val].format(col=col)
                else:
                    groupby_['"{}"'.format(name)] = col
        elif isinstance(groupby, (list, tuple)) and len(groupby) > 0:
            groupby_ = {
                '"{}"'.format(key.replace('"', '').replace(self.quote, '')): key if '"' in key or self.quote in key
                else '"t"."{}"'.format(key)
                for key in groupby
            }
        elif isinstance(groupby, str) and len(groupby) > 0:
            groupby_ = {
                '"{}"'.format(groupby.replace('"', '').replace(self.quote, '')): groupby if '"' in groupby or self.quote in groupby
                else '"t"."{}"'.format(groupby)
            }

        query = 'SELECT'
        if len(groupby_) > 0:
            query += '\n    {},'.format(',\n    '.join([
                '{} AS {}'.format(val, key) for key, val in groupby_.items()
            ]))
        query += '\n    {}'.format(',\n    '.join([
            '{} AS {}'.format(val, key) for key, val in columns_.items()
        ]))
        query += '\nFROM {} AS "t"'.format(table)

        if isinstance(filters, (list, tuple)) and len(filters) > 0:
            query += '\nWHERE {}'.format('\n    AND '.join(filters))

        if len(groupby_) > 0:
            query += '\nGROUP BY {}'.format(', '.join(list(groupby_.values())))

        if limit:
            query += '\nLIMIT {:d}'.format(limit)

        if offset:
            query += '\nOFFSET {:d}'.format(offset)
        
        return query

    def get_results(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> pd.DataFrame:
        """
        Get results from the table.

        Parameters
        ----------
        table : str, optional
            Table name.
            If not provided, the query must be a SELECT string query.
        query : str or dict, optional
            Query string or dictionary.
        
        Returns
        -------
        pd.DataFrame
            Results from the table.
        """
        query = query or {}

        if table is None and not isinstance(query, str):
            return pd.DataFrame()
        elif table is not None and isinstance(query, dict):
            query = self.build_select_query(table, **query)

        result = self.execute(query)

        if result is None:
            return pd.DataFrame()

        columns = result.keys()
        data = result.fetchall()
        result.close()

        return pd.DataFrame.from_records(data, columns=columns, coerce_float=True)
    
    def get_row(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> dict:
        """
        Get a single row from the table.

        Parameters
        ----------
        table : str, optional
            Table name.
            If not provided, the query must be a SELECT string query.
        query : str or dict, optional
            Query string or dictionary.
        
        Returns
        -------
        dict
            Single row from the table.
        """
        query = query or {}

        if table is None and not isinstance(query, str):
            return {}
        elif table is not None and isinstance(query, dict):
            query['limit'] = 1
            query = self.build_select_query(table, **query)

        result = self.execute(query)

        if result is None:
            return {}

        columns = result.keys()
        data = result.fetchone()
        result.close()

        if data is None or len(data) == 0:
            return {}

        return dict(zip(columns, data))
    
    def get_var(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> Any:
        """
        Get a single value from the table.

        Parameters
        ----------
        table : str, optional
            Table name.
            If not provided, the query must be a SELECT string query.
        query : str or dict, optional
            Query string or dictionary.
        
        Returns
        -------
        Any
            Single value from the table.
        """
        query = query or {}

        if table is None and not isinstance(query, str):
            return False
        elif table is not None and isinstance(query, dict):
            query['limit'] = 1
            query = self.build_select_query(table, **query)

        result = self.execute(query)

        if result is None:
            return False

        data = result.fetchone()
        result.close()

        if data is None or len(data) == 0:
            return

        return data[0]
    
    def get_agg(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> Any:
        """
        Get an aggregated value from the table.

        Parameters
        ----------
        table : str, optional
            Table name.
            If not provided, the query must be a SELECT string query.
        query : str or dict, optional
            Query string or dictionary.
        
        Returns
        -------
        Any
            Aggregated value from the table.
        """
        query = query or {}

        if table is None and not isinstance(query, str):
            return
        elif table is not None and isinstance(query, dict):
            query = self.build_agg_query(table, **query)

        agg = self.get_results(table, query)

        if agg.shape == (1, 1):
            agg = agg.values[0][0]

        return agg
    
    def get_count(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> int:
        """
        Get the count of rows in the table.

        Parameters
        ----------
        table : str, optional
            Table name.
            If not provided, the query must be a SELECT string query.
        query : str or dict, optional
            Query string or dictionary.
        
        Returns
        -------
        int
            Count of rows in the table.
        """
        return self.get_agg(table, query)

    def insert(
        self,
        table: str,
        query: Optional[pd.DataFrame | str] = None,
        truncate: bool = True,
        max_rows: int = 100000,
        **kwargs
    ) -> int:
        """
        Insert data into the table.

        Parameters
        ----------
        table : str
            Table name.
        query : pd.DataFrame or str, optional
            Data to insert into the table.
            If a DataFrame is provided, data will be formatted and inserted into the table in chunks.
            If a string is provided, it will be executed as an INSERT query.
            If not provided, a load/copy query will be executed if the stage file exists.
        truncate : bool, optional
            If True, truncate the table before inserting the data.
        max_rows : int, optional
            Maximum number of rows to insert in a single query.
            Only applicable when `query` is a DataFrame.
        kwargs : dict
            Additional arguments to be passed to the load/copy query.
        
        Returns
        -------
        int
            Number of rows inserted into the table.
        """
        if truncate:
            self.truncate(table)

        if isinstance(query, pd.DataFrame):
            data = query.copy()
            maxins = self.params.get('maxins') or max_rows

            chunks = int(np.ceil(data.shape[0] / maxins))
            for i in np.arange(chunks):
                if chunks > 1:
                    print('{}/{}...'.format(i + 1, chunks))

                start = i * maxins
                end = start + maxins
                query = self.build_insert_query(table, data.iloc[start:end])
                self.execute(query)
        elif isinstance(query, str):
            query = 'INSERT INTO {} {}'.format(table, query)
            self.execute(query)
        else:
            query, path = self.build_load_query(table, **kwargs)
            if query is not None:
                self.execute_load(query, path)

        rows = self.get_agg(table)

        return rows

    def upsert(
        self,
        table: str,
        query: Optional[pd.DataFrame | str] = None,
        key: Optional[str | list[str]] = None,
        max_rows: int = 100000,
        **kwargs
    ) -> int:
        """
        Upsert data into the table. This method will insert the data into a the table,
        and if a row with the same key already exists, it will update the existing row.

        Parameters
        ----------
        table : str
            Table name.
        query : pd.DataFrame or str, optional
            Data to upsert into the table.
            If a DataFrame is provided, data will be formatted and upserted into the table in chunks.
            If a string is provided, it will be executed as an UPSERT query.
            If not provided, a load/copy query will be executed if the stage file exists.
        key : str or list[str], optional
            Key column(s) to use for upserting data.
            If not provided, the first column in the DataFrame will be used.
        max_rows : int, optional
            Maximum number of rows to upsert in a single query.
            Only applicable when `query` is a DataFrame.
        kwargs : dict
            Additional arguments to be passed to the load/copy query.
        
        Returns
        -------
        int
            Number of rows upserted into the table.
        """
        if isinstance(query, pd.DataFrame):
            data = query.copy()
            rows = data.shape[0]
            maxins = self.params.get('maxins') or max_rows

            chunks = int(np.ceil(data.shape[0] / maxins))
            for i in np.arange(chunks):
                if chunks > 1:
                    print('{}/{}...'.format(i + 1, chunks))

                start = i * maxins
                end = start + maxins
                query = self.build_upsert_query(table, data.iloc[start:end], key=key)
                self.execute(query)
        else:
            stage = '{}__stage'.format(table)

            if 'stage' not in kwargs:
                kwargs['stage'] = self.get_table_name(table)

            self.clone(table, stage).insert(stage, query, **kwargs)

            rows = self.get_agg(stage)

            filters = []
            if isinstance(key, (list, tuple)):
                for i in key:
                    filters.append('{}."{}" = {}."{}"'.format(table, i, stage, i))
            else:
                filters.append('{}."{}" = {}."{}"'.format(table, key, stage, key))

            self.execute('DELETE FROM {} USING {} WHERE {}'.format(table, stage, ' AND '.join(filters)))
            self.execute('INSERT INTO {} SELECT * FROM {}'.format(table, stage))

            self.drop(stage)

        return rows
    
    def unload(
        self,
        table: str,
        query: Optional[str | dict] = None,
        **kwargs
    ) -> str:
        """
        Unload data from the table.

        Parameters
        ----------
        table : str
            Table name.
        query : str or dict, optional
            Query string or dictionary.
        kwargs : dict
            Additional arguments to be passed to the unload/copy query.
        
        Returns
        -------
        str
            Path to the file where the data was unloaded.
        """
        query = query or {}

        if isinstance(query, dict):
            query = self.build_select_query(table, **query)

        query, path = self.build_unload_query(table, query, **kwargs)
        self.execute_unload(query, path)

        return path

    def create(
        self,
        table: str,
        meta: dict,
        key: Optional[str | list[str]] = None,
        relations: Optional[dict] = None,
        indexes: Optional[dict] = None,
        autokey: bool = False,
        sortkey: Optional[str | list[str]] = None,
        diststyle: Optional[str | list[str]] = None,
        distkey: Optional[str | list[str]] = None,
        temp: bool = False,
        replace: bool = True
    ) -> Self:
        """
        Create a table in the database.

        Parameters
        ----------
        table : str
            Table name.
        meta : dict
            Table schema definitions.
        relations : dict, optional
            Table relations definitions.
        indexes : dict, optional
            Table indexes definitions.
        autokey : bool, optional
            If True, set the primary key as an AUTOINCREMENT key.
        sortkey : str or list[str], optional
            Column(s) to be used for sorting.
        diststyle : str or list[str], optional
            Distribution style parameter.
        distkey : str or list[str], optional
            Distribution key parameter.
        temp : bool, optional
            If True, set the table as a temporary table.
        replace : bool, optional
            If True, replace the table if it already exists.
        """
        if replace:
            self.drop(table)

        query = self.build_create_query(
            table,
            meta=meta,
            key=key,
            relations=relations,
            indexes=indexes,
            autokey=autokey,
            sortkey=sortkey,
            diststyle=diststyle,
            distkey=distkey,
            temp=temp
        )
        
        self.execute(query, autocommit=True)

        return self

    def modify(
        self,
        table: str,
        meta: dict,
        key: Optional[str | list[str]] = None,
        relations: Optional[dict] = None,
        indexes: Optional[dict] = None,
        autokey: bool = False,
        sortkey: Optional[str | list[str]] = None,
        diststyle: Optional[str | list[str]] = None,
        distkey: Optional[str | list[str]] = None,
        vfill: Optional[str | dict] = None
    ) -> Self:
        """
        Modify a table schema by adding or removing columns.

        Parameters
        ----------
        table : str
            Table name.
        meta : dict
            Table schema definitions.
        key : str or list[str], optional
            Column(s) to be used as primary key.
        relations : dict, optional
            Table relations definitions.
        indexes : dict, optional
            Table indexes definitions.
        autokey : bool, optional
            If True, set the primary key as an AUTOINCREMENT key.
        sortkey : str or list[str], optional
            Column(s) to be used for sorting.
        diststyle : str or list[str], optional
            Distribution style parameter.
        distkey : str or list[str], optional
            Distribution key parameter.
        vfill : str or dict, optional
            Value(s) to fill in the new columns.
            If a string is provided, it will be used to fill all new columns.
            If a dictionary is provided, it will be used to fill specific columns.
        """
        name = self.get_table_name(table)
        tmp_name = '{}__temp'.format(table)

        current_columns = self.get_columns(table)
        new_columns = list(meta)

        # For each column in the new table, fill with the existing column value if it exists,
        # otherwise fill with the provided value in `vfill` or NULL if it is not provided
        select = ', '.join([
            '"{}"'.format(col) if col in current_columns
            else '{} AS "{}"'.format(vfill[col], col) if isinstance(vfill, dict) and col in vfill
            else '{} AS "{}"'.format(vfill, col) if isinstance(vfill, str)
            else 'NULL AS "{}"'.format(col)
            for col in new_columns
        ])

        # Create a new table with the modified schema
        self.create(
            tmp_name,
            meta=meta,
            key=key,
            relations=relations,
            indexes=indexes,
            autokey=autokey,
            sortkey=sortkey,
            diststyle=diststyle,
            distkey=distkey
        )

        # Copy data from the current table to the new table
        self.execute('INSERT INTO {} SELECT {} FROM {}'.format(tmp_name, select, table))

        # Drop the current table and rename the new one
        self.drop(table)
        self.rename(tmp_name, name)

        return self

    def clone(
        self,
        table: str,
        name: str,
        temp: bool = False,
        replace: bool = True
    ) -> Self:
        """
        Clone a table in the database.

        Parameters
        ----------
        table : str
            Table name to clone.
        name : str
            New table name.
        temp : bool, optional
            If True, set the table as a temporary table.
        replace : bool, optional
            If True, replace the table if it already exists.
        """
        if replace:
            self.drop(name)

        query = self.build_clone_query(
            table,
            name,
            temp=temp
        )

        self.execute(query, autocommit=True)

        return self
    
    def get_version(
        self
    ) -> str:
        """
        Get the database server version.
        """
        return
    
    def get_tables(
        self,
        schema: Optional[str] = None
    ) -> list[str]:
        """
        Get the database schema tables.

        Parameters
        ----------
        schema : str, optional
            Schema name to get tables from.
            If not provided, the default schema will be used.
        
        Returns
        -------
        list[str]
            List of table names.
        """
        return
    
    def get_columns(
        self,
        table: str
    ) -> list[str]:
        """
        Get the database table columns.

        Parameters
        ----------
        table : str
            Table name to get columns from.
        
        Returns
        -------
        list[str]
            List of column names.
        """
        return
    
    def get_meta(
        self,
        table: str
    ) -> dict[str, dict]:
        """
        Get the database table metadata.

        Parameters
        ----------
        table : str
            Table name to get metadata from.
        
        Returns
        -------
        dict[str, dict]
            Table metadata as a dictionary.
        """
        return

    def truncate(
        self,
        table: str
    ) -> Self:
        """
        Truncate a table in the database.

        Parameters
        ----------
        table : str
            Table name to truncate.
        """
        self.execute(
            'TRUNCATE {}'.format(table),
            autocommit=True
        )

        return self

    def drop(
        self,
        table: str
    ) -> Self:
        """
        Drop a table in the database.

        Parameters
        ----------
        table : str
            Table name to drop.
        """
        self.execute(
            'DROP TABLE IF EXISTS {} CASCADE'.format(table),
            autocommit=True
        )

        return self

    def rename(
        self,
        table: str,
        name: str
    ) -> Self:
        """
        Rename a table in the database.

        Parameters
        ----------
        table : str
            Table name to rename.
        name : str
            New table name.
        """
        self.execute(
            'ALTER TABLE {} RENAME TO {}'.format(table, name),
            autocommit=True
        )

        return self

    def reset_autokey(
        self,
        table: str,
        key: str
    ) -> Self:
        """
        Reset the AUTOINCREMENT key for a table.

        Parameters
        ----------
        table : str
            Table name to reset.
        key : str
            Column to reset the AUTOINCREMENT key for.
        """
        return self

    def vacuum(
        self,
        table: str
    ) -> Self:
        """
        Vacuum a table in the database.

        Parameters
        ----------
        table : str
            Table name to vacuum.
        """
        return self


class Model(Core):
    """
    Base class for data models. Provides methods for interacting with source and target database tables.
    Each model represents a table in a database and should be defined as a subclass of this class.

    Attributes
    ----------
    database : str, optional
        Database this model belongs to.
        If not provided, the model will be associated with the default database.
    table : str, optional
        The table that this model represents in the database.
        If not provided, the model will not be associated with a database table.
    key : str
        Column to be used as primary key.
    sort : str
        Column to be used for sorting.
    autokey : bool
        If True, the primary key will be set as an AUTOINCREMENT key.
    update_col : str, optional
        Column to be used to determine the last update timestamp.
    meta : dict or str, optional
        Table schema definitions.
        If a string is provided, it should be a JSON string or a path to a JSON file.
    relations : dict or str, optional
        Table relations definitions.
        If a string is provided, it should be a JSON string or a path to a JSON file.
    indexes : dict or str, optional
        Table indexes definitions.
        If a string is provided, it should be a JSON string or a path to a JSON file.
    diststyle : str, optional
        Distribution style parameter.
    distkey : str, optional
        Distribution key parameter.
    """

    database = None
    table = None

    key = 'id'
    sort = 'id'
    autokey = False
    update_col = None

    meta = None
    relations = None
    indexes = None

    diststyle = None
    distkey = None

    def __init__(self) -> None:
        if type(self) is Model:
            raise TypeError('Abstract class should not be instantiated.')

        super().__init__()

        # Get the database adapter for this model (if table is provided)
        self._dal = self._get_dal()

        # Build table schema from provided definitions
        self.meta = self._build_meta(self.meta)
        self.relations = self._build_relations(self.relations)
        self.indexes = self._build_indexes(self.indexes)

        # Autokey is not supported for composite keys
        if self.autokey and not isinstance(self.key, str):
            self.autokey = False
    
    def _get_dal(self) -> Adapter:
        """
        Get the database adapter for this model.
        """
        # If table is not provided, the model is not associated with a database table
        if self.table is not None:
            # If database is not provided, use the default database
            if self.database is None:
                return self.app.db
            else:
                return self.app.dbs[self.database]
    
    def _build_meta(
        self,
        meta: Optional[dict] = None
    ) -> dict:
        """
        Build the table schema from the provided meta definition.
        """
        if is_string(meta) and self.app.fs.exists(meta):
            meta = self.app.fs.read(meta)

        if is_json(meta):
            meta = json.loads(meta)
        
        if is_empty(meta) or not isinstance(meta, dict):
            return
        
        return parse_meta(meta)
    
    def _build_relations(
        self,
        relations: Optional[dict] = None
    ) -> dict:
        """
        Build the table relations from the provided relations definition.
        """
        if is_string(relations) and self.app.fs.exists(relations):
            relations = self.app.fs.read(relations)

        if is_json(relations):
            relations = json.loads(relations)
        
        if is_empty(relations) or not isinstance(relations, dict):
            return

        for key, rel in relations.items():
            if rel.get('table') is None and rel.get('model') is not None:
                if '.' in rel['model']:
                    module, name = rel['model'].split('.')
                else:
                    module = self.__class__.__module__.rsplit('.', 1)[-1]
                    name = rel['model']

                namespace = import_module('...models.{}'.format(module), package=__name__)
                model = getattr(namespace, name)()

                relations[key]['table'] = model.table
                relations[key]['column'] = model.key

        return relations
    
    def _build_indexes(
        self,
        indexes: Optional[dict] = None
    ) -> dict:
        """
        Build the table indexes from the provided indexes definition.
        """
        if is_string(indexes) and self.app.fs.exists(indexes):
            indexes = self.app.fs.read(indexes)

        if is_json(indexes):
            indexes = json.loads(indexes)
        
        if is_empty(indexes) or not isinstance(indexes, dict):
            return

        return indexes
    
    @property
    def columns(self) -> list[str]:
        return list(self.meta)
    
    @property
    def stage(self) -> bool:
        if self._dal is not None:
            return self._dal.stage
        
        return False

    def begin(self):
        """
        Begin a transaction session.
        """
        self._dal.begin()

        return self

    def commit(self):
        """
        Commit the current transaction session.
        """
        self._dal.commit()

        return self

    def rollback(self):
        """
        Rollback the current transaction session.
        """
        self._dal.rollback()

        return self
    
    def get_db_columns(self) -> list[str]:
        """
        Get the columns of the current table in the database.

        Returns
        -------
        list[str]
            List of column names in the table.
        """
        return self._dal.get_columns(self.table)
    
    def get_db_meta(self) -> dict[str, dict]:
        """
        Get the schema of the current table in the database.

        Returns
        -------
        dict[str, dict]
            Table schema definitions as a meta dictionary.
        """
        return self._dal.get_meta(self.table)
    
    def execute(
        self,
        query: str,
        **kwargs
    ) -> Any:
        """
        Execute a query in the database.

        Parameters
        ----------
        query : str
            SQL query to execute.
        kwargs : dict
            Additional arguments to be passed to the adapter's execute method.
        
        Returns
        -------
        Any
            Result of the query execution.
        """
        return self._dal.execute(query, **kwargs)

    def get_results(
        self,
        query: Optional[str | dict] = None,
        formatted: bool = False
    ) -> pd.DataFrame:
        """
        Get results from the table.

        Parameters
        ----------
        query : str or dict, optional
            Query string or dictionary.
        formatted : bool
            If True, format the data before returning.
        
        Returns
        -------
        pd.DataFrame
            Results from the table.
        """
        query = query or {}

        if isinstance(query, dict):
            if 'columns' not in query:
                query['columns'] = list(self.meta.keys())
            if 'sort' not in query:
                query['sort'] = self.sort

        df = self._dal.get_results(self.table, query)

        if formatted:
            relations = query['relations'] if 'relations' in query else None
            df = self.format_data(df, relations=relations)

        return df

    def get_row(
        self,
        query: Optional[str | dict] = None
    ) -> dict:
        """
        Get a single row from the table.

        Parameters
        ----------
        query : str or dict, optional
            Query string or dictionary.
        
        Returns
        -------
        dict
            Single row from the table.
        """
        query = query or {}

        if isinstance(query, dict):
            if 'columns' not in query:
                query['columns'] = list(self.meta.keys())
            if 'sort' not in query:
                query['sort'] = self.sort

        return self._dal.get_row(self.table, query)

    def get_var(
        self,
        query: Optional[str | dict] = None
    ) -> Any:
        """
        Get a single value from the table.

        Parameters
        ----------
        query : str or dict, optional
            Query string or dictionary.
        
        Returns
        -------
        Any
            Single value from the table.
        """
        return self._dal.get_var(self.table, query)

    def get_agg(
        self,
        query: Optional[str | dict] = None
    ) -> Any:
        """
        Get an aggregated value from the table.

        Parameters
        ----------
        query : str or dict, optional
            Query string or dictionary.
        
        Returns
        -------
        Any
            Aggregated value from the table.
        """
        return self._dal.get_agg(self.table, query)

    def get_count(
        self,
        query: Optional[str | dict] = None
    ) -> int:
        """
        Get the count of rows in the table.

        Parameters
        ----------
        query : str or dict, optional
            Query string or dictionary.
        
        Returns
        -------
        int
            Count of rows in the table.
        """
        return self._dal.get_count(self.table, query)

    def get_update_ts(
        self,
        update_col: Optional[str] = None,
        filters: Optional[list] = None
    ) -> pd.Timestamp | int:
        """
        Get the Timestamp or ID of the last update in the table.

        Parameters
        ----------
        update_col : str, optional
            Name of the column to look at for the last update.
        filters : list, optional
            List of filters to apply to the query
        
        Returns
        -------
        pd.Timestamp | int
            Timestamp or ID of the last update in the table.
        """
        query = {
            'columns': update_col or self.update_col,
            'agg': 'max',
            'filters': filters or []
        }

        update_ts = self.get_agg(query)

        if is_datetime(update_ts):
            update_ts = pd.Timestamp(str(update_ts))
        elif is_number(update_ts):
            update_ts = int(update_ts)

        return update_ts

    def get_results_staged(
        self,
        query: Optional[str | dict] = None,
        formatted: bool = False,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Unload data from the table to the stage directory and read it into a DataFrame.
        This is a faster method to get results, as it reads all files unloaded in paralell using Dask,
        then concatenates them into a single DataFrame, finally cleaning the stage directory.

        Parameters
        ----------
        query : str or dict, optional
            Query string or dictionary.
        formatted : bool
            If True, format the data before returning.
        compression : str, optional
            Compression type for the stage file.
        kwargs : dict
            Additional arguments to be passed to the adapter's stage_read method.
        
        Returns
        -------
        pd.DataFrame
            Data read from the stage file.
        """
        if not self.stage:
            return

        path = self.unload(self.table, query, **kwargs)

        path += '*'
        if compression == 'gzip':
            path += '.gz'

        meta = {col: object for col in self._meta}

        df = (
            dd
            .read_csv(
                path,
                storage_options={
                    'key': self.app.config.aws.key,
                    'secret': self.app.config.aws.secret
                },
                compression=compression,
                header=None,
                names=list(meta.keys()),
                dtype=meta
            )
            .compute()
        )

        self._dal.stage_clean(self.get_table_name())

        if formatted:
            relations = query['relations'] if 'relations' in query else None
            df = self.format_data(df, relations=relations)

        return df

    def insert(
        self,
        query: Optional[pd.DataFrame | str] = None,
        **kwargs
    ) -> int:
        """
        Insert data into the table.

        Parameters
        ----------
        query : pd.DataFrame or str, optional
            Data to insert into the table.
            If a DataFrame is provided, data will be formatted and inserted into the table in chunks.
            If a string is provided, it will be executed as an INSERT query.
            If not provided, a load/copy query will be executed if the stage file exists.
        kwargs : dict
            Additional arguments to be passed to the adapter's insert method.
        
        Returns
        -------
        int
            Number of rows inserted into the table.
        """
        if query is None and self.stage and not self.stage_exists():
            return 0
        
        if isinstance(query, pd.DataFrame):
            query = self.format_query(query)

        return self._dal.insert(self.table, query, **kwargs)

    def upsert(
        self,
        query: Optional[pd.DataFrame | str] = None,
        **kwargs
    ) -> int:
        """
        Upsert data into the table. This method will insert the data into the table,
        and if a row with the same key already exists, it will update the existing row.

        Parameters
        ----------
        query : pd.DataFrame or str, optional
            Data to upsert into the table.
            If a DataFrame is provided, data will be formatted and upserted into the table in chunks.
            If a string is provided, it will be executed as an UPSERT query.
            If not provided, a load/copy query will be executed if the stage file exists.
        kwargs : dict
            Additional arguments to be passed to the adapter's upsert method.
        
        Returns
        -------
        int
            Number of rows upserted into the table.
        """
        if query is None and self.stage and not self.stage_exists():
            return 0
        
        if isinstance(query, pd.DataFrame):
            query = self.format_query(query)

        return self._dal.upsert(self.table, query, key=self.key, **kwargs)
    
    def unload(
        self,
        query: Optional[str | dict] = None,
        **kwargs
    ) -> str:
        """
        Unload data from the table to a file in the stage directory.

        Parameters
        ----------
        query : str or dict, optional
            Query string or dictionary.
        kwargs : dict
            Additional arguments to be passed to the adapter's unload method.
        
        Returns
        -------
        str
            Path to the file where the data was unloaded.
        """
        if not self.stage:
            return

        query = query or {}

        if isinstance(query, dict):
            if 'sort' not in query:
                query['sort'] = self.sort

        return self._dal.unload(self.table, query, **kwargs)

    def create(
        self,
        temp: bool = False,
        replace: bool = True
    ) -> Self:
        """
        Create the table in the database.

        Parameters
        ----------
        temp : bool
            If True, set the table as a temporary table.
        replace : bool
            If True, replace the table if it already exists.
        """
        self._dal.create(
            self.table,
            meta=self.meta,
            key=self.key,
            relations=self.relations,
            indexes=self.indexes,
            autokey=self.autokey,
            sortkey=self.sort,
            diststyle=self.diststyle,
            distkey=self.distkey,
            temp=temp,
            replace=replace
        )

        return self

    def modify(
        self,
        vfill: Optional[str | dict] = None
    ) -> Self:
        """
        Modify a table schema by adding or removing columns.

        Parameters
        ----------
        vfill : str or dict, optional
            Value(s) to fill in the new columns.
            If a string is provided, it will be used to fill all new columns.
            If a dictionary is provided, it will be used to fill specific columns.
        """
        self._dal.modify(
            self.table,
            meta=self.meta,
            key=self.key,
            relations=self.relations,
            indexes=self.indexes,
            autokey=self.autokey,
            sortkey=self.sort,
            diststyle=self.diststyle,
            distkey=self.distkey,
            vfill=vfill
        )

        return self

    def truncate(self) -> Self:
        """
        Truncate the table in the database.
        """
        self._dal.truncate(self.table)

        return self

    def drop(self) -> Self:
        """
        Drop the table in the database.
        """
        self._dal.drop(self.table)

        return self

    def rename(self, name: str) -> Self:
        """
        Rename the table in the database.

        Parameters
        ----------
        name : str
            New table name.
        """
        self._dal.rename(self.table, name)

        return self

    def reset_autokey(self):
        """
        Reset the AUTOINCREMENT key for the table.
        """
        if self.autokey:
            self._dal.reset_autokey(self.table, self.key)

        return self

    def vacuum(self) -> Self:
        """
        Vacuum the table in the database, in order to reclaim storage
        and reorganize the physical storage of the table.
        """
        self._dal.vacuum(self.table)

        return self

    def stage_write(
        self,
        df: pd.DataFrame,
        chunk: Optional[str] = None,
        compression: Optional[Literal['gzip']] = None,
    ) -> str:
        """
        Write data to the stage directory.

        Parameters
        ----------
        df : pd.DataFrame
            Data to write to the stage file.
        chunk : str, optional
            Chunk identifier for the file.
        compression : str, optional
            Compression type for the stage file.
        
        Returns
        -------
        str
            Path to the file where the data was written.
        """
        if not self.stage:
            return

        name = self.get_stage_path(chunk=chunk, compression=compression)

        data = self.format_query(df)

        return self.to_csv(data, name, compression=compression, header=False)

    def stage_exists(self) -> bool:
        """
        Check if the stage file exists.
        """
        if not self.stage:
            return False

        name = '/'.join(self.get_stage_path().split('/')[:-1])

        return self.app.fs.exists(name)

    def stage_clean(self) -> Self:
        """
        Clean the stage directory.
        """
        if not self.stage:
            return self

        name = '/'.join(self.get_stage_path().split('/')[:-1])

        if self.app.fs.exists(name):
            self.app.fs.remove(name)

        return self
    
    def get_table_parts(self) -> tuple[str, str]:
        """
        Split table name into schema and name components.

        Returns
        -------
        tuple[str, str]
            Schema and name parts of the table name.
        """
        if '.' in self.table:
            schema, name = self.table.split('.', 1)
        else:
            schema = None
            name = self.table

        return schema, name

    def get_table_name(self) -> str:
        """
        Get only the name of the table excluding its schema component.

        Returns
        -------
        str
            Name component of the table name.
        """
        _, name = self.get_table_parts()

        return name

    def get_stage_path(
        self,
        chunk: Optional[str] = None,
        compression: Optional[Literal['gzip']] = None
    ) -> str:
        """
        Get the path to the stage file.
        """
        if not self.stage:
            return

        table = self.get_table_name()

        path = '{}/{}/{}'.format(self._dal.stage_dir, table, table)

        if chunk is not None:
            path += '_' + chunk

        path += '.csv'

        if compression == 'gzip':
            path += '.gz'

        return path

    def to_csv(
        self,
        df: pd.DataFrame,
        name: str,
        sep: str = ',',
        index: bool = False,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> str:
        """
        Write data to a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            Data to write to the CSV file.
        name : str
            Name of the CSV file.
        sep : str
            String of length 1. Field delimiter for the output file.
        index : bool
            Write row names (index).
            See the pandas.to_csv method for more information.
        compression : str, optional
            Compression type for the CSV file.
        kwargs : dict
            Additional arguments to be passed to the FileSystem's write_csv method.
        
        Returns
        -------
        str
            Path to the file where the data was written.
        """
        return self.app.fs.write_csv(
            df,
            name,
            sep=sep,
            index=index,
            compression=compression,
            **kwargs
        )

    def read_csv(
        self,
        name: str,
        sep: str = ',',
        compression: Optional[Literal['gzip']] = None,
        formatted: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read data from a CSV file.

        Parameters
        ----------
        name : str
            Name of the CSV file.
        sep : str
            String of length 1. Field delimiter for the output file.
        compression : str, optional
            Compression type for the CSV file.
        formatted : bool
            If True, format the data before returning.
        kwargs : dict
            Additional arguments to be passed to the FileSystem's read_csv method.
        
        Returns
        -------
        pd.DataFrame
            Data read from the CSV file.
        """
        df = self.app.fs.read_csv(
            name,
            sep=sep,
            compression=compression,
            **kwargs
        )

        if formatted:
            df = self.format_data(df)

        return df

    def to_json(
        self,
        df: pd.DataFrame,
        name: str,
        orient: str = 'records',
        date_format: Optional[Literal['epoch', 'iso']] = 'iso',
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ):
        """
        Write data to a JSON file.

        Parameters
        ----------
        df : pd.DataFrame
            Data to write to the JSON file.
        name : str
            Name of the JSON file.
        orient : str
            Indication of expected JSON string format.
            See the pandas.to_json method for more information.
        date_format : str
            Type of date conversion.
            If None, the default date format is used.
            See the pandas.to_json method for more information.
        compression : str, optional
            Compression type for the JSON file.
        kwargs : dict
            Additional arguments to be passed to the FileSystem's write_json method.
        
        Returns
        -------
        str
            Path to the file where the data was written.
        """
        return self.app.fs.write_json(
            df,
            name,
            orient=orient,
            date_format=date_format,
            compression=compression,
            **kwargs
        )

    def read_json(
        self,
        name: str,
        orient: Literal['records'] = 'records',
        compression: Optional[Literal['gzip']] = None,
        formatted: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read data from a JSON file.

        Parameters
        ----------
        name : str
            Name of the JSON file.
        orient : str
            Indication of expected JSON string format.
            See the pandas.read_json method for more information.
        compression : str, optional
            Compression type for the JSON file.
        formatted : bool
            If True, format the data before returning.
        kwargs : dict
            Additional arguments to be passed to the FileSystem's read_json method.
        
        Returns
        -------
        pd.DataFrame
            Data read from the JSON file.
        """
        df = self.app.fs.read_json(
            name,
            orient=orient,
            compression=compression,
            **kwargs
        )

        if formatted:
            df = self.format_data(df)

        return df

    def build_df(self) -> pd.DataFrame:
        """
        Build an empty DataFrame from the table schema.

        Returns
        -------
        pd.DataFrame
            Empty DataFrame with all the columns in the table.
        """
        return format_data(None, self.meta)

    def format_data(
        self,
        df: pd.DataFrame,
        index: bool = False,
        sort: bool = False,
        int_type: Literal['numeric', 'nullable', 'strict'] = 'numeric',
        bin_type: Literal['bool', 'category', 'numeric', 'strict'] = 'bool',
        obj_type: Literal['obj', 'json'] = 'obj',
        prevent_nulls: bool = False,
        relations: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Format a DataFrame in order to match column descriptions,
        optimizing each column's type to its lowest memory size.

        Parameters
        ----------
        df : pd.DataFrame
            Data to be formattted.
        index : bool
            If True, set the model's key as index of the DataFrame.
        sort : bool
            If True, sort the columns of the DataFrame by model's sort column(s).
        int_type : {'numeric', 'nullable', 'strict'}, default 'numeric'
            How to handle integer columns.

            - ``numeric`` :
                Try to convert to 'int' dtype if there are no null values or 'float' instead.
                This is better if we need to operate on data, allowing optimization of memory and modules communication.
            - ``nullable`` :
                Set dtype to Pandas 'Int', which allows null values in its integer type.
                Needed if we are performing database operations with nullable integer fields, preventing unhelping errors.
            - ``strict`` :
                Fill missing values and force 'int' dtype.
        bin_type : {'bool', 'category', 'numeric', 'strict'}, default 'bool'
            How to handle binary columns.

            - ``bool`` :
                Set dtype to 'bool' (default), setting missing values to False.
            - ``category`` :
                Set dtype to 'category' (True | False), useful for some data operations and visualization.
            - ``numeric`` :
                Treat column as a numeric value (1 or 0), in order to fit into numeric matrices and training tensors.
            - ``nullable`` :
                Set dtype to Pandas 'Int', which allows null values in its integer type.
                Needed if we are performing database operations with nullable integer fields, preventing unhelping errors.
            - ``strict`` :
                Fill missing values and force 'bool' dtype.
        obj_type : {'obj', 'json'}, default 'obj'
            How to handle object columns.

            - ``obj`` :
                Set dtype to Python object (default).
            - ``json`` :
                Set dtype to JSON string.
        prevent_nulls : bool, default False
            If True, fill missing values with their column default value (defined in ``meta`` dictionary for each column).
        relations : list, optional
            List of relations to include in the formatted data.
        
        Returns
        -------
        pd.DataFrame
            Formatted data.
        """
        relations = relations or []

        meta = self.meta
        cols = df.columns

        for rel in relations:
            for key, item in rel['model'].meta.items():
                if rel['name'] + '__' + key in cols:
                    meta[rel['name'] + '__' + key] = item

        return format_data(
            df,
            meta=meta,
            index=self.key if index else None,
            sort=self.sort if sort else None,
            int_type=int_type,
            bin_type=bin_type,
            obj_type=obj_type,
            prevent_nulls=prevent_nulls
        )

    def format_query(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format SQL query before executing an INSERT or UPSERT operation.

        Parameters
        ----------
        df : pd.DataFrame
            Data to be formattted.
        
        Returns
        -------
        pd.DataFrame
            Formatted data.
        """
        data = df.copy()

        for col, params in self.meta.items():
            if col not in data.columns:
                continue

            dtype = params['type']
            if dtype == 'dts':
                data[col] = data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                data[col] = data[col].where(data[col] != 'NaT', None)
            elif dtype == 'dtd':
                data[col] = data[col].dt.strftime('%Y-%m-%d')
                data[col] = data[col].where(data[col] != 'NaT', None)

        if self.autokey:
            if data[self.key].isnull().all():
                data[self.key] = 'DEFAULT'

        return data
