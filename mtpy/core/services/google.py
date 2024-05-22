from datetime import datetime
import numpy as np
import pandas as pd
from google.auth.credentials import Credentials as GoogleCredentials
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build as GoogleClient
from google.analytics.data_v1beta import BetaAnalyticsDataClient as GoogleAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
from google.ads.googleads.client import GoogleAdsClient

from typing import Callable, Optional, Self

from ..data import Adapter
from ..io import FileSystem
from ..utils.dates import Format
from ..utils.helpers import is_array, is_empty, is_string


class GoogleService(Adapter):

    service: str | Callable = ''
    scope: list | str = []
    default_version: Optional[str] = None

    def __init__(self, params: Optional[dict] = None):
        if type(self) is GoogleService:
            raise TypeError('Abstract class should not be instantiated.')

        super().__init__(params=params)

    def _build_client(self) -> Self:
        version = self.params.get('version', self.default_version)
        credentials = self._get_credentials()

        if is_string(self.service):
            args = dict(
                serviceName=self.service,
                credentials=credentials
            )
            if version is not None:
                args['version'] = version

            self._client = GoogleClient(**args)
        elif callable(self.service):
            args = dict(
                credentials=credentials
            )
            if version is not None:
                args['version'] = version

            self._client = self.service(**args)

        return self
    
    def _get_credentials(self) -> GoogleCredentials:
        if is_string(self.scope):
            scopes = [f'https://www.googleapis.com/auth/{self.scope}.readonly']
        elif is_array(self.scope):
            scopes = [f'https://www.googleapis.com/auth/{scope}.readonly' for scope in self.scope]
        else:
            scopes = None

        credentials = self.params.get('credentials')

        if is_string(credentials):
            filename = FileSystem.assure_local_file(credentials)
            return ServiceAccountCredentials.from_json_keyfile_name(
                filename,
                scopes=scopes
            )
        elif isinstance(credentials, dict):
            return ServiceAccountCredentials.from_json_keyfile_dict(
                credentials,
                scopes=scopes
            )
        else:
            raise ValueError('Invalid credentials format.')

class GoogleSheets(GoogleService):

    service = 'sheets'
    scope = ['spreadsheets']
    default_version = 'v4'

    def get_results(
        self,
        sheet_id: str,
        range: str
    ) -> pd.DataFrame:
        result = []

        response = self._client.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=range
        ).execute()

        values = response.get('values', [])

        columns = values[0]
        rows = values[1:]

        for row in rows:
            if len(row) < len(columns):
                row += list(np.repeat('', len(columns) - len(row)))

            result.append(row)

        df = pd.DataFrame(
            result,
            columns=columns
        )

        return df

class Youtube(GoogleService):

    service = 'youtube'
    scope = ['youtube']
    default_version = 'v3'

    def list_channel_videos(
        self,
        id: str,
        order: Optional[str] = 'date',
        limit: Optional[int] = 1000
    ) -> list:
        self.build()

        response = self._client.search().list(
            part='id',
            channelId=id,
            maxResults=limit,
            order=order,
            type='video'
        ).execute()

        result = []

        for item in response.get('items', []):
            result.append(
                item.get('id', {}).get('videoId')
            )

        return result
    
    def get_videos(
        self,
        ids: Optional[str | list] = None,
        channel_id: Optional[str] = None,
        order: Optional[str] = 'date',
        limit: Optional[int] = 1000,
        ret_stats: Optional[bool] = False
    ) -> pd.DataFrame:
        self.build()

        if is_empty(ids):
            if not is_empty(channel_id):
                ids = self.list_channel_videos(
                    channel_id,
                    order=order,
                    limit=limit
                )
            else:
                raise ValueError('Either ids or channel_id must be provided.')
        elif is_string(ids):
            ids = [ids]

        parts = ['snippet', 'contentDetails']
        if ret_stats:
            parts.append('statistics')

        response = self._client.videos().list(
            part=','.join(parts),
            id=','.join(ids)
        ).execute()

        result = []

        for item in response.get('items', []):
            row = {
                'id': item.get('id'),
                'published_at': item.get('snippet', {}).get('publishedAt'),
                'title': item.get('snippet', {}).get('title'),
                'description': item.get('snippet', {}).get('description'),
                'tags': item.get('snippet', {}).get('tags'),
                'category_id': item.get('snippet', {}).get('categoryId'),
                'duration': item.get('contentDetails', {}).get('duration')
            }

            if ret_stats:
                row |= {
                    'num_views': item.get('statistics', {}).get('viewCount'),
                    'num_likes': item.get('statistics', {}).get('likeCount'),
                    'num_dislikes': item.get('statistics', {}).get('dislikeCount'),
                    'num_favorites': item.get('statistics', {}).get('favoriteCount'),
                    'num_comments': item.get('statistics', {}).get('commentCount')
                }

            result.append(row)

        df = pd.DataFrame(result)
        df['published_at'] = pd.to_datetime(df.published_at, utc=True).dt.tz_localize(None)

        return df


class GoogleAnalytics(GoogleService):

    service = 'analyticsreporting'
    scope = ['analytics']
    default_version = 'v4'

    def get_report(
        self,
        dt_from: datetime | str,
        dt_to: datetime | str,
        dimensions: Optional[list] = [],
        metrics: Optional[list] = [],
        limit: Optional[int] = 1000
    ) -> pd.DataFrame:
        self.build()

        result = []

        dt_from = pd.Timestamp(dt_from)
        dt_to = pd.Timestamp(dt_to)

        page_token = 0
        while True:
            response = self._client.reports().batchGet(
                body={
                    'reportRequests': [
                        {
                            'viewId': self.params['view_id'],
                            'dateRanges': [
                                {
                                    'startDate': dt_from.strftime(Format.SQLD),
                                    'endDate': dt_to.strftime(Format.SQLD)
                                }
                            ],
                            'dimensions': [{'name': 'ga:{}'.format(i)} for i in dimensions],
                            'metrics': [{'expression': 'ga:{}'.format(i)} for i in metrics],
                            'pageToken': str(page_token),
                            'pageSize': str(limit)
                        }
                    ]
                }
            ).execute()

            report = response['reports'][0]
            result.extend([
                row['dimensions'] + row['metrics'][0]['values']
                for row in report['data']['rows']
            ])

            if 'nextPageToken' in report:
                page_token = report['nextPageToken']
            else:
                break

        df = pd.DataFrame(
            result,
            columns=dimensions + metrics
        ).sort_values(dimensions).reset_index(drop=True)
        df['date'] = pd.to_datetime(df.date)

        return df


class GoogleAnalyticsData(GoogleService):

    service = GoogleAnalyticsDataClient
    scope = 'analytics'
    default_version = None  # 'v16'

    def _build_client(self) -> Self:
        credentials = self.params.get('credentials')

        if is_string(credentials):
            filename = FileSystem.assure_local_file(credentials)
            self._client = GoogleAnalyticsDataClient.from_service_account_file(
                filename
            )
        elif isinstance(credentials, dict):
            self._client = GoogleAnalyticsDataClient.from_service_account_info(
                credentials
            )
        else:
            raise ValueError('Invalid credentials format.')
        
        return self

    def get_report(
        self,
        dt_from: datetime | str,
        dt_to: datetime | str,
        dimensions: Optional[list] = [],
        metrics: Optional[list] = [],
        limit: Optional[int] = 1000
    ) -> pd.DataFrame:
        self.build()

        result = []

        dt_from = pd.Timestamp(dt_from)
        dt_to = pd.Timestamp(dt_to)

        response = self._client.run_report(
            RunReportRequest(
                property='properties/{}'.format(self.params['property_id']),
                date_ranges=[
                    DateRange(
                        start_date=dt_from.strftime(Format.SQLD),
                        end_date=dt_to.strftime(Format.SQLD)
                    )
                ],
                dimensions=[
                    Dimension(name=i) for i in dimensions
                ],
                metrics=[
                    Metric(name=i) for i in metrics
                ],
                limit=limit
            )
        )

        for row in response.rows:
            r = [
                i.value for i in row.dimension_values
            ] + [
                i.value for i in row.metric_values
            ]

            result.append(r)

        df = pd.DataFrame(
            result,
            columns=dimensions + metrics
        ).sort_values(dimensions).reset_index(drop=True)
        df['date'] = pd.to_datetime(df.date)

        return df


class GoogleAds(GoogleService):

    service = GoogleAdsClient
    scope = ['adwords']
    default_version = 'v16'

    def get_results(
        self,
        dt_from: datetime | str,
        dt_to: datetime | str
    ) -> pd.DataFrame:
        result = []

        dt_from = pd.Timestamp(dt_from)
        dt_to = pd.Timestamp(dt_to)

        service = self._client.get_service('GoogleAdsService')

        query = """
            SELECT
                segments.date,
                campaign.id,
                campaign.name,
                ad_group.id,
                ad_group.name,
                metrics.cost_micros,
                metrics.clicks,
                metrics.all_conversions,
                metrics.conversions_value
            FROM ad_group
            WHERE segments.date BETWEEN '{dt_from}' AND '{dt_to}'
            ORDER BY segments.date, campaign.id, ad_group.id
        """.format(
            dt_from=dt_from.strftime(Format.SQLD),
            dt_to=dt_to.strftime(Format.SQLD)
        )

        request = self.client.get_type('SearchGoogleAdsStreamRequest')
        request.customer_id = self.params['customer_id']
        request.query = query

        response = service.search_stream(request)

        for batch in response:
            for row in batch.results:
                r = [
                    row.segments.date,
                    row.campaign.id,
                    row.campaign.name,
                    row.ad_group.id,
                    row.ad_group.name,
                    row.metrics.cost_micros,
                    row.metrics.clicks,
                    row.metrics.all_conversions,
                    row.metrics.conversions_value
                ]

                result.append(r)

        df = pd.DataFrame(
            result,
            columns=[
                'date', 'campaign_id', 'campaign', 'group_id', 'group',
                'cost', 'clicks', 'conversions', 'value'
            ]
        )

        if df.shape[0] > 0:
            df['date'] = pd.to_datetime(df.date)
            df['cost'] = df.cost / 1000000
            df = df.sort_values(['date', 'campaign_id', 'group_id'], ignore_index=True)

        return df
