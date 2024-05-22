import pandas as pd
from simple_salesforce import Salesforce as SalesforceClient

from typing import Any, Optional
from typing_extensions import Self

from ..data import DBAdapter


class Salesforce(DBAdapter):

    stypes_map = {
        'id': 'str',
        'boolean': 'bin',
        'reference': 'str',
        'string': 'str',
        'picklist': 'cat',
        'combobox': 'cat',
        'textarea': 'str',
        'double': 'num',
        'address': ['obj', 2048],
        'phone': 'str',
        'url': 'str',
        'currency': 'num',
        'int': 'int',
        'datetime': 'dts',
        'date': 'dtd',
        'email': 'str',
        'multipicklist': 'cat',
        'percent': 'num',
        'encryptedstring': 'str'
    }

    def _build_client(self) -> Self:
        self._client = SalesforceClient(**self.params)

        return self

    def _execute_client(self, query: str) -> Any:
        self.build()

        return self._client.query_all(query)
    
    def describe(self, table: Optional[str] = None):
        self.build()

        cursor = getattr(self._client, table) if table is not None else self._client

        try:
            result = cursor.describe()
        except Exception as e:
            raise e

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
        if isinstance(columns, str):
            columns = columns.split(',')
        if isinstance(columns, (list, tuple)) and len(columns) > 0:
            select = '{}'.format(', '.join(columns))
        else:
            select = 'FIELDS(ALL)'
        
        query = 'SELECT {} FROM {}'.format(select, table)

        if isinstance(filters, (list, tuple)) and len(filters) > 0:
            query += ' WHERE {}'.format(' AND '.join(filters))

        if isinstance(sort, str):
            sort = sort.split(',')
        if isinstance(sort, (list, tuple)) and len(sort) > 0:
            sort = [
                '{} {}'.format(*s.split(' ')) if ' ' in s
                else '{}'.format(s)
                for s in sort
            ]
            query += ' ORDER BY {}'.format(', '.join(sort))

        if limit:
            query += ' LIMIT {:d}'.format(limit)

        if offset:
            query += ' OFFSET {:d}'.format(offset)

        return query
    
    def get_row(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> dict:
        query = query or {}

        if table is not None and isinstance(query, dict):
            query['limit'] = 1
            query = self.build_select_query(table, **query)

        result = self.execute(query)

        if result is None:
            return {}

        df = pd.DataFrame(result['records'])

        if 'attributes' in df.columns:
            df = df.drop(columns=['attributes'])

        if df.shape[0] == 0:
            return {}

        return df.iloc[0].to_dict()

    def get_results(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> pd.DataFrame:
        query = query or {}

        if table is not None and isinstance(query, dict):
            query = self.build_select_query(table, **query)

        result = self.execute(query)

        if result is None:
            return pd.DataFrame()

        df = pd.DataFrame(result['records'])

        if 'attributes' in df.columns:
            df = df.drop(columns=['attributes'])

        return df
    
    def get_tables(
        self,
        schema: Optional[str] = None
    ) -> list[str]:
        return [
            x['name'] for x in self.describe()['sobjects']
        ]
    
    def get_columns(
        self,
        table: str
    ) -> list[str]:
        return [
            x['name'] for x in self.describe(table)['fields']
        ]
    
    def get_meta(
        self,
        table: str
    ) -> dict[str, dict]:
        sfields = {
            x['name']: {
                'type': x['type'],
                'length': x['length']
            } for x in self.describe(table)['fields']
        }

        meta = {}
        for name, desc in sfields.items():
            value = self.stypes_map.get(desc['type'])

            if isinstance(value, str) and value in ['str', 'obj', 'cat'] and desc['length'] > 0:
                value = [value, desc['length']]
            
            meta[name] = value

        return meta
