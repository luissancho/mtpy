from datetime import datetime
import pandas as pd

from typing import Optional

from ..data import APIAdapter
from ..utils.dates import Format, is_datetime
from ..utils.helpers import is_empty


class Simplecast(APIAdapter):
    """
    API adapter for Simplecast, a podcast hosting and distribution platform.

    API Doc: https://apidocs.simplecast.com

    Attributes
    ----------
    api_url : str
        The base URL for the Simplecast API.
    """

    api_url = 'https://api.simplecast.com'

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params=params)

        self.headers = {
            'Authorization': f"Bearer {self.params['token']}"
        }

    def get_episodes(
        self,
        podcast_id: Optional[str] = None,
        search: Optional[str] = None,
        status: Optional[str] = None,
        type: Optional[str] = None,
        order: Optional[str] = None,
        limit: Optional[int] = 1000,
        offset: Optional[int] = 0,
        ret_stats: Optional[bool] = False
    ) -> pd.DataFrame:
        """
        Get data of all the episodes of a given podcast.

        Parameters
        ----------
        podcast_id : str
            The ID of the podcast.
            If not provided, it will search for this value in the adapter's params.
        search : str, optional
            Search for episodes by title.
        status : str, optional
            Filter episodes by status.
            Available statuses are:
                `importing`, `audio_imported`, `transcoding`, `transcoding_error`,
                `draft`, `scheduled`, `published`, `private`.
        type : str, optional
            Filter episodes by type.
            Available types are:
                `full`, `trailer`, `bonus`.
        order : str, optional
            Sorts the episodes that are returned.
            Available sorting fields are:
                `title`, `number`, `custom_url`, `published_at`, `created_at` or `updated_at`.
            Sort order is ascending by default and can be changed by appending `_asc` or `_desc` to a field.
            For example, to sort by title descending you'd use `title_desc`.
        limit : int, optional
            Number of records per page.
        offset : int, optional
            Record offset for pagination.
        ret_stats : bool, optional
            Whether to include download stats.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame.
        
        Raises
        ------
        ValueError
            If `podcast_id` is not provided.
        """
        if is_empty(podcast_id):
            if 'podcast_id' in self.params:
                podcast_id = self.params['podcast_id']
            else:
                raise ValueError('Param <podcast_id> must be provided.')

        endpoint = f'podcasts/{podcast_id}/episodes'

        data = {
            'type': 'full',
            'search': search,
            'status': status,
            'type': type,
            'sort': order,
            'limit': limit,
            'offset': offset
        }

        response = self.execute(
            method='GET',
            endpoint=endpoint,
            data=data,
            headers=self.headers
        )

        result = []

        for item in response['collection']:
            result.append({
                'id': item.get('id'),
                'published_at': item.get('published_at'),
                'title': item.get('title'),
                'description': item.get('description'),
                'duration': item.get('duration')
            })

        df = pd.DataFrame(result)
        df['published_at'] = pd.to_datetime(df.published_at, utc=True).dt.tz_localize(None)

        if ret_stats:
            downloads = self.get_downloads(
                podcast_id=podcast_id,
                order=order,
                limit=limit
            )
            df = df.merge(downloads, on='id', how='left')

        return df

    def get_downloads(
        self,
        podcast_id: Optional[str] = None,
        dt_from: Optional[datetime | str] = None,
        dt_to: Optional[datetime | str] = None,
        order: Optional[str] = None,
        limit: Optional[int] = 1000
    ) -> pd.DataFrame:
        """
        Get download stats for a given podcast, grouped by Episode.

        Parameters
        ----------
        podcast_id : str
            The ID of the podcast.
            If not provided, it will search for this value in the adapter's params.
        dt_from : datetime | str, optional, default `published_at`
            Date from which to get the data.
            Defaults to the `published_at` date
        dt_to : datetime | str, optional, default `now`
            Date until which to get the data.
            Defaults to now.
        order : str, optional
            Sorts analytics by date of interval.
            Available sort options are:
                `asc`, `desc`.
        limit : int, optional
            Number of records per page.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame.
        
        Raises
        ------
        ValueError
            If `podcast_id` is not provided.
        """
        if is_empty(podcast_id):
            if 'podcast_id' in self.params:
                podcast_id = self.params['podcast_id']
            else:
                raise ValueError('Param <podcast_id> must be provided.')

        endpoint = 'analytics/episodes'

        params = {
            'podcast': podcast_id,
            'sort': order,
            'limit': limit
        }
        
        if is_datetime(dt_from):
            params['start_date'] = pd.Timestamp(dt_from).strftime(Format.SQLD)
        if is_datetime(dt_to):
            params['end_date'] = pd.Timestamp(dt_to).strftime(Format.SQLD)

        response = self.execute(
            method='GET',
            endpoint=endpoint,
            params=params,
            headers=self.headers
        )

        result = []

        for item in response.get('collection', []):
            result.append({
                'id': item.get('id'),
                'num_downloads': item.get('downloads', {}).get('total')
            })

        df = pd.DataFrame(result)

        return df
    
    def get_analytics_group(
        self,
        name: str,
        episode_id: Optional[str] = None,
        podcast_id: Optional[str] = None,
        dt_from: Optional[datetime | str] = None,
        dt_to: Optional[datetime | str] = None
    ) -> pd.DataFrame:
        """
        Get analytics collected for a given Podcast or Episode, grouped by a specific metric group.
        Either `episode_id` or `podcast_id` must be provided.

        Parameters
        ----------
        name : str
            The name of the group to get the stats split by.
            Available groups are:
                `location`, `applications`, `operating_systems`, `devices`, `device_class`,
                `browsers`, `providers`, `network_types`, `listening_methods`.
        episode_id : str, optional
            The ID of the episode.
        podcast_id : str, optional
            The ID of the podcast.
            If not provided, it will search for this value in the adapter's params.
        dt_from : datetime | str, optional
            Date from which to get the data.
        dt_to : datetime | str, optional
            Date until which to get the data.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame.
        
        Raises
        ------
        ValueError
            If `podcast_id` is not provided.
        """
        if is_empty(episode_id):
            if is_empty(podcast_id):
                if 'podcast_id' in self.params:
                    podcast_id = self.params['podcast_id']
                else:
                    raise ValueError('Param <podcast_id> must be provided.')

        if name == 'location':
            endpoint = f'analytics/{name}'
        else:
            endpoint = f'analytics/technology/{name}'
        
        params = {}

        if not is_empty(episode_id):
            params['episode'] = episode_id
        else:
            params['podcast'] = podcast_id
        
        if is_datetime(dt_from):
            params['start_date'] = pd.Timestamp(dt_from).strftime(Format.SQLD)
        if is_datetime(dt_to):
            params['end_date'] = pd.Timestamp(dt_to).strftime(Format.SQLD)

        response = self.execute(
            method='GET',
            endpoint=endpoint,
            params=params,
            headers=self.headers
        )

        result = []

        if name == 'location':
            rows = response.get('countries', [])
        else:
            rows = response.get('collection', [])

        for item in rows:
            result.append({
                'rank': item.get('rank'),
                'name': item.get('name'),
                'num_downloads': item.get('downloads_total')
            })

        df = pd.DataFrame(result)

        return df
    
    def get_analytics_groups(
        self,
        names: list,
        episode_id: Optional[str] = None,
        podcast_id: Optional[str] = None,
        dt_from: Optional[datetime | str] = None,
        dt_to: Optional[datetime | str] = None
    ) -> pd.DataFrame:
        """
        Get analytics collected for a given Podcast or Episode, grouped by a specific metric group.
        Either `episode_id` or `podcast_id` must be provided.

        Parameters
        ----------
        names : list
            A list of group names to get the stats split by.
            Available groups are:
                `location`, `applications`, `operating_systems`, `devices`, `device_class`,
                `browsers`, `providers`, `network_types`, `listening_methods`.
        episode_id : str, optional
            The ID of the episode.
        podcast_id : str, optional
            The ID of the podcast.
            If not provided, it will search for this value in the adapter's params.
        dt_from : datetime | str, optional
            Date from which to get the data.
        dt_to : datetime | str, optional
            Date until which to get the data.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame.
        
        Raises
        ------
        ValueError
            If `podcast_id` is not provided.
        """
        if is_empty(episode_id):
            if is_empty(podcast_id):
                if 'podcast_id' in self.params:
                    podcast_id = self.params['podcast_id']
                else:
                    raise ValueError('Param <podcast_id> must be provided.')

        columns = ['group', 'name', 'num_downloads']
        df = pd.DataFrame(columns=columns)

        for name in names:
            d = self.get_analytics_group(
                name,
                episode_id=episode_id,
                podcast_id=podcast_id,
                dt_from=dt_from,
                dt_to=dt_to
            )
            d['group'] = name

            df = pd.concat([df, d[columns]], ignore_index=True)
        
        return df
    
    def get_analytics_series(
        self,
        name: str,
        podcast_id: Optional[str] = None,
        freq: Optional[str] = 'day',
        dt_from: Optional[datetime | str] = None,
        dt_to: Optional[datetime | str] = None
    ) -> pd.DataFrame:
        """
        Get analytics collected for a given Podcast, grouped by a specific metric series.

        Parameters
        ----------
        name : str
            The name of the series to get the stats split by.
        podcast_id : str
            The ID of the podcast.
            If not provided, it will search for this value in the adapter's params.
        freq : str, optional, default `day`
            Splits the download analytics up by the interval given.
            Available intervals are:
                `day`, `week`, `month`.
        dt_from : datetime | str, optional
            Date from which to get the data.
        dt_to : datetime | str, optional
            Date until which to get the data.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame.
        
        Raises
        ------
        ValueError
            If `podcast_id` is not provided.
        """
        if is_empty(podcast_id):
            if 'podcast_id' in self.params:
                podcast_id = self.params['podcast_id']
            else:
                raise ValueError('Param <podcast_id> must be provided.')

        endpoint = f'analytics/{name}'

        params = {
            'podcast': podcast_id,
            'interval': freq
        }
        
        if is_datetime(dt_from):
            params['start_date'] = pd.Timestamp(dt_from).strftime(Format.SQLD)
        if is_datetime(dt_to):
            params['end_date'] = pd.Timestamp(dt_to).strftime(Format.SQLD)

        response = self.execute(
            method='GET',
            endpoint=endpoint,
            params=params,
            headers=self.headers
        )

        result = []

        for item in response.get('by_interval', []):
            result.append({
                'date': item.get('interval'),
                f'num_{name}': item.get(f'{name}_total')
            })

        df = pd.DataFrame(result)

        return df
