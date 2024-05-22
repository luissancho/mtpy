import json
import pandas as pd
from pymongo import MongoClient

from typing import Any, Optional
from typing_extensions import Self

from ..data import DBAdapter
from ..io import FileSystem


class Mongo(DBAdapter):

    def _build_client(self) -> Self:
        db_params = self.params.copy()

        if 'ssl_ca_certs' in db_params:
            db_params['ssl_ca_certs'] = FileSystem.assure_local_file(db_params['ssl_ca_certs'])

        if '://' not in db_params['host']:
            db_params['host'] = 'mongodb://{}'.format(db_params['host'])

        self._client = MongoClient(**db_params).get_database(self.database)

        return self
    
    def names(self):
        self.build()

        result = self._client.collection_names()

        return result

    def exists(self, table):
        result = bool(table in self.names())

        return result

    def select(self, table):
        if self.exists(table):
            return self._client[table]

    def count(self, table, query=None):
        query = query or {}

        return self.select(table).count(**query)

    def find(self, table, query=None):
        query = query or {}

        docs = self.select(table).find(**query)

        return pd.DataFrame(list(docs))

    def aggregate(self, table, pipeline):
        docs = self.select(table).aggregate(pipeline)

        return list(docs)

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
        query = {
            'filter': None,
            'sort': None,
            'limit': 0,
            'skip': 0
        }

        if isinstance(filters, dict) and len(filters) > 0:
            query['filter'] = filters

        if isinstance(sort, str):
            sort = sort.split(',')
        if isinstance(sort, (list, tuple)) and len(sort) > 0:
            query['sort'] = [(i, 1) if i[0] != '-' else (i[1:], -1) for i in sort]

        if limit:
            query['limit'] = limit

        if offset:
            query['skip'] = offset

        return query

    def get_results(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> pd.DataFrame:
        query = query or {}

        columns = query.get('columns', [])
        if isinstance(columns, str):
            columns = columns.split(',')
        
        query = self.build_select_query(table, **query)

        result = self.find(table, query)

        if len(columns) > 0:
            columns = [col for col in columns if col in result.columns]

            if len(columns) == 0:
                return pd.DataFrame()

            result = result[columns]

        return result

    def get_row(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> dict:
        query = query or {}

        columns = query.get('columns', [])
        if isinstance(columns, str):
            columns = columns.split(',')

        query['limit'] = 1

        query = self.build_select_query(table, **query)

        result = self.find(table, query)

        if result.shape[0] == 0:
            return {}

        if len(columns) > 0:
            columns = [col for col in columns if col in result.columns]

            if len(columns) == 0:
                return {}

            result = result[columns]

        return result.iloc[0].to_dict()
    
    def get_var(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> Any:
        query = query or {}

        columns = query.get('columns', [])
        if isinstance(columns, str):
            columns = columns.split(',')

        query['limit'] = 1

        query = self.build_select_query(table, **query)

        result = self.find(table, query)

        if result.shape[0] == 0:
            return

        if len(columns) > 0:
            columns = [col for col in columns if col in result.columns]

            if len(columns) == 0:
                return

            result = result[columns]

        return result[result.columns[0]].iloc[0]
    
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
        query = query or {}

        query = self.build_select_query(table, **query)

        return self.count(table, query)

    def insert(
        self,
        table: str,
        query: Optional[pd.DataFrame | str] = None,
        truncate: bool = True
    ) -> int:
        if truncate:
            self.truncate(table)

        if isinstance(query, pd.DataFrame):
            query = query.to_dict(orient='records')
        elif isinstance(query, str):
            query = json.loads(query)
        elif not isinstance(query, dict):
            return 0

        self.select(table).insert_many(query)

        rows = self.count(table)

        return rows

    def create(
        self,
        table: str,
        replace: bool = True
    ) -> Self:
        if self.exists(table):
            if replace:
                self.drop(table)
            else:
                return self

        self.dbh.create_collection(table)

        return self
    
    def get_tables(
        self,
        schema: Optional[str] = None
    ) -> list[str]:
        return self.names()

    def truncate(
        self,
        table: str
    ) -> Self:
        self.select(table).delete_many({})

        return self

    def drop(
        self,
        table: str
    ) -> Self:
        self.select(table).drop()

        return self
