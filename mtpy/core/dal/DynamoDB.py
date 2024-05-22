import boto3
from boto3.dynamodb.types import TypeDeserializer
import pandas as pd

from typing import Any, Optional
from typing_extensions import Self

from ..data import DBAdapter


class DynamoDB(DBAdapter):

    def _build_client(self) -> Self:
        self._client = boto3.client('dynamodb', **self.params)

        return self

    def _execute_client(self, query: Any, page_token: str = None) -> Any:
        payload = dict(Statement=query)
        if page_token is not None:
            payload |= dict(NextToken=page_token)

        return self._client.execute_statement(**payload)

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
            select = '{}'.format(','.join(columns))
        else:
            select = '*'
        
        query = 'SELECT {} FROM "{}"'.format(select, table)

        if isinstance(filters, (list, tuple)) and len(filters) > 0:
            query += ' WHERE {}'.format(' AND '.join(filters))

        return query

    def get_results(
        self,
        table: Optional[str] = None,
        query: Optional[str | dict] = None
    ) -> pd.DataFrame:
        query = query or {}

        if table is not None and isinstance(query, dict):
            query = self.build_select_query(table, **query)

        result = []
        page_token = None

        while True:
            r = self.execute(query, page_token=page_token)

            td = TypeDeserializer()
            result.extend([{
                k: str(td.deserialize(v)) for k, v in item.items()
            } for item in r.get('Items', [])])

            if 'NextToken' in r:
                page_token = r['NextToken']
            else:
                break

        df = pd.DataFrame(result)

        if 'sort' in query:
            df = df.sort_values(query['sort'])

        return df
    
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

        td = TypeDeserializer()
        df = pd.DataFrame([{
            k: str(td.deserialize(v)) for k, v in item.items()
        } for item in result.get('Items', [])])

        if df.shape[0] == 0:
            return {}

        return df.iloc[0].to_dict()
