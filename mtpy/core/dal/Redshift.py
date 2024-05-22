from typing import Literal, Optional
from typing_extensions import Self

from ..data import SQLAdapter


class Redshift(SQLAdapter):

    driver = 'redshift+psycopg2'
    default_port = 5439
    quote = '"'
    data_types = {
        'varchar': 'VARCHAR',
        'text': 'TEXT',
        'date': 'DATE',
        'datetime': 'TIMESTAMP',
        'numeric': 'DECIMAL',
        'tinyint': 'SMALLINT',
        'smallint': 'SMALLINT',
        'mediumint': 'INTEGER',
        'integer': 'INTEGER',
        'bigint': 'BIGINT',
        'boolean': 'BOOLEAN'
    }
    create_suffix = 'BACKUP Yes DISTSTYLE {diststyle} DISTKEY ({distkey}) SORTKEY ({sortkey})'

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
        aws_params = kwargs.get('aws_params', self.params)

        if stage is None:
            stage = self.get_table_name(table)
        
        path = '{}/{}/{}'.format(self.stage_dir, stage, stage)

        cmd = [
            'COPY {table}',
            "FROM 's3://{bucket}/{path}'",
            "ACCESS_KEY_ID '{key}'",
            "SECRET_ACCESS_KEY '{secret}'",
            "REGION '{region}'",
            'FORMAT {fmt}',
            "DELIMITER '{sep}'",
            'EMPTYASNULL',
            "NULL AS 'None'",
            'ROUNDEC',
            'COMPUPDATE OFF',
            'STATUPDATE OFF'
        ]

        if compression == 'gzip':
            cmd.append('GZIP')

        query = '\n'.join(cmd).format(
            table=table,
            bucket=aws_params['bucket'],
            path=path,
            key=aws_params['key'],
            secret=aws_params['secret'],
            region=aws_params['region'],
            fmt=fmt.upper(),
            sep=sep
        )

        return query, path

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
        parallel = kwargs.get('parallel', False)
        max_size = kwargs.get('max_size', '100 MB')
        aws_params = kwargs.get('aws_params', self.params)

        if stage is None:
            stage = self.get_table_name(table)
        
        path = '{}/{}/{}'.format(self.stage_dir, stage, stage)

        cmd = [
            "UNLOAD ('{query}')",
            "TO 's3://{bucket}/{path}'",
            "ACCESS_KEY_ID '{key}'",
            "SECRET_ACCESS_KEY '{secret}'",
            "REGION '{region}'",
            'FORMAT {fmt}',
            "NULL AS ''",
            'ALLOWOVERWRITE'
        ]

        if parallel:
            cmd.append('MAXFILESIZE {max_size}')
        else:
            cmd.append('PARALLEL OFF')

        if fmt != 'parquet':
            cmd.append("DELIMITER '{sep}'")

            if header:
                cmd.append('HEADER')

            if compression == 'gzip':
                cmd.append('GZIP')

        query = '\n'.join(cmd).format(
            query=query.replace("'", r"\'"),
            bucket=aws_params['bucket'],
            path=path,
            key=aws_params['key'],
            secret=aws_params['secret'],
            region=aws_params['region'],
            fmt=fmt.upper(),
            sep=sep,
            max_size=max_size
        )

        return query, path
    
    def get_version(
        self
    ) -> str:
        return self.get_var(query='SELECT VERSION()')
    
    def get_tables(
        self,
        schema: Optional[str] = None
    ) -> list[str]:
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = '{}'".format(schema or 'public')
        
        tables = self.get_results(query=query)['table_name'].tolist()

        return tables
    
    def get_columns(
        self,
        table: str
    ) -> list[str]:
        schema, name = self.get_table_parts(table)
        
        query = "SELECT d.column FROM pg_table_def AS d WHERE d.tablename = '{}'".format(name)
        if schema is not None:
            query += " AND d.schemaname = '{}'".format(schema)
        
        columns = self.get_results(query=query)['column'].tolist()

        return columns
    
    def get_meta(
        self,
        table: str
    ) -> dict[str, dict]:
        meta = {
            col: {} for col in self.get_columns(table)
        }

        return meta

    def vacuum(
        self,
        table: str
    ) -> Self:
        self.execute(
            'VACUUM FULL {} TO 99 PERCENT'.format(table),
            autocommit=True
        )
        self.execute(
            'ANALYZE {}'.format(table),
            autocommit=True
        )

        return self
