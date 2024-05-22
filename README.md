# MTpy ETL Framework

## Description

MTpy is a Python framework that provides a simple and intuitive interface for defining and running data pipelines. It is designed to be automatically deployed to the cloud using Docker.

## Fundamental Concepts

### Pipelines

A pipeline is a sequence of data processing steps. It is a series of tasks that are executed in a specific order. Each pipeline has a unique name and can be used to extract, transform, and load (ETL) data from different sources to different targets. It`s process sequence is defined by a dictionary with the following structure:

```python
pipes = {
    'action1': {
        'Setup': [`task1`, `task2`, ...],
        'Extract': [`task3`, `task4`, ...],
        'Transform': [`task5`, `task6`, ...],
        'Load': [`task7`, `task8`, ...],
        'Build': [`task9`, `task10`, ...],
        'Set': [`task11`, `task12`, ...]
    },
    'action2': {
        ...
    },
    ...
}
```

It is a dictionary with 3 levels of nesting, where the first level is the action to be executed, the second level is the stage of the action (Setup, Extract, Transform, Load, Build, Set), and the third level is the list of tasks to be executed in each of the stages.

Tipical actions are:
    - `update`: perform an incremental update of the target model.
    - `bulk`: read all data from source, transform it and load it into target.
    - `rebulk`: read all data from source, transform it and upsert it into target.
    - `load`: load data from stage file into target.
    - `unload`: unload data from target into stage file.
    ...

The tasks within each stage must be defined as object methods with `_task` suffix:

```python
def task1_task(self):
    # Code to execute the task
    return self
```

A set of main default tasks (e.g. `read`, `insert`, `upsert`, `stage`, `vacuum`, etc...) are already defined and can be used in the pipelines, but they can be overridden in the subclass.

Also, for extracting and loading data, a source and target models must be defined:

```python
source: = 'SourceModel'
target: = 'TargetModel'
```

A pipeline can be executed in the following manner:

```python
python /path/to/project/pipeline.py pipeline_name update
```

This will run all the stages of the `PipelineName` pipeline within the `update` action.

### Jobs

A job is a process that can be executed programmatically via cron scheduling, from a worker queue listener, an API call or manually using the CLI.

Specifically, a SequenceJob is a job that runs a sequence of pipelines, which is defined by a dictionary with the following structure:

```python
sequence = {
    'group1': [
        'pipeline1',
        'pipeline2',
        ...
    ],
    'group2': [
        ...
    ],
    ...
}
```

A job can be executed in the following manner:

```python
python /path/to/project/job.py job_name update
```

This will run all the pipelines of the `JobName` job, executing all the stages within the `update` action of each pipeline.

### Models

A model represents a table in a database. All pipelines are assigned a `source` model from which to extract information and a `target` model where to load it once transformed. Each model contains the following attributes:

    - `database`: The name of the database where the table is located.
    - `table`: Table name.
    - `key`: Primary key.
    - `sort`: Sorting key.
    - `autokey`: Whether the primary key will be set as an AUTOINCREMENT key or not.
    - `update_col`: The column that will be used as last update Timestamp or ID.
    - `meta`: Table schema definition.
    ...

Every model has a "meta" attribute, which is a schema dictionary with column descriptions. These are the parameters needed to define each of the columns:

    - ``type``:
        Data type, which consists in any of the folowing:
            - `str`: cast value as string (default length 256).
            - `obj`: for objects, arrays or any json-type values (default length 256).
            - `cat`: categorize value as a `category` dtype (default length 256).
            - `dtd`: cast value as datetime and keep only the date part (default length 4).
            - `dts`: cast value as datetime with date and time parts (default length 8).
            - `num`: cast value as float (default length [16, 2]).
            - `int`: cast value as integer (default length 4).
            - `bin`: cast value as binary or boolean-type (default length 1).
    - ``length``:
        Maximum length of the value (in bytes), needed for database table definition.
    - ``nullable``:
        If True, this column will accept null values, needed for database table definition (default True).
    - ``default``:
        Default value of column, in order to fill missing values if needed (default None).

Example:

```python
meta = {
    'col1': {'type': 'str', 'length': 256, 'nullable': True, 'default': None},  # As a dict
    'col2': ['str', 256, True, None],  # As a list of parameters
    'col3': ['int', 4, True, 'mean'],  # Use column 'mean' function as default value
    'col4': ['num', [16, 2]]  # Numeric type needs definition of integer and decimal parts
    'col5': 'dtd'  # Just the column type (keeping parameters defaults),
    'col6': 'varchar'  # Alias may be used instead of the original type name
}
```

## Usage

In order to create a new Pipeline, you need to create both the source and target models, the pipeline itself and a job that will be scheduled to run it.

Suppose we want to create a new pipeline to extract data from a source database, make a series of transformations and load the resulting data into a target database. The source data is stored in a MySQL database table called `users` and the target data is stored in a PostgreSQL database table called `website.users` (table name is `users` and schema name is `website`).

### Create the source model

Open a new file in `mtpy/models/sources` folder with name `website.py` and define the `Users` class with the following attributes:

```python
from ...core.data import Model

class Users(Model):

    database = 'src_dbname'
    table = 'users'
    key = 'id'
    sort = 'created_at'
    
    meta = {
        'id': 'int',
        'created_at': 'dts',
        'updated_at': 'dts',
        'name': 'str'
    }
```

Save the file and we will have a new source model ready to be used in our pipelines.

### Create the target model

Open a new file in `mtpy/models` folder with name `website.py` and define the `Users` class with the following structure:

```python
from ..core.data import Model

class Users(Model):

    database = 'tgt_dbname'
    table = 'website.users'
    key = 'id'
    sort = 'created_at'
    update_col = 'updated_at'
    
    meta = {
        'id': 'int',
        'created_at': 'dts',
        'updated_at': 'dts',
        'name': 'str'
    }
```

Save the file and we will have a new target model ready to be used in our pipelines.

### Create the pipeline

Open a new file in `mtpy/pipelines/website` folder with name `Users.py` and define the `Users` class with the following attributes:

```python
from ..core.worker import Pipeline
from ..models.sources.website import Users as mSource
from ..models.website import Users as mTarget

class Users(Pipeline):

    pipes = {
        'bulk': {
            'Extract': ['read'],
            'Transform': ['transform'],
            'Load': ['insert']
        }
    }

    source = mSource
    target = mTarget

    def transform_task(self):
        df = self.data

        # Transform data here

        self.data = df

        return self
```

The `read_task` and `insert_task` methods don't need to be declared because they are already defined in the `Pipeline` base class.

### Create the job

Open a new file in `mtpy/jobs` folder with name `Website.py` and define the `Website` class with the following structure:

```python
from ..core.worker import SequenceJob


class Website(SequenceJob):

    sequence = {
        'website': [
            'users'
        ]
    }
    default_action = 'bulk'
```

To run this job, you can use the following command:

```bash
python /path/to/project/job.py website
```

This will run all the pipelines of the `website` job, executing all the stages within the `bulk` action of each pipeline.

## Deployment

MTpy is designed to be deployed to the cloud using Docker. The following steps will guide you through the process of deploying MTpy to the cloud.

### Docker image build and run

In order to build a Docker image of the MTpy project, you need to run the following command in the project root folder:

```bash
docker build -f ./Dockerfile -t mtpy .
```

This will create a Docker image of the MTpy project with the tag `mtpy`.

Then, you can run the Docker image with the following command:

```bash
docker run -d -p 8042:8042 --env-file=.env mtpy
```

This will run the MTpy Docker image in a detached mode, exposing the API on port 8042.

### Configuration

The `.env` file contains the environment variables needed to run the MTpy project, such as the database connection strings, the RabbitMQ broker URL, etc. The following environment variables must be set in order to run the framework:

- `APP_ENV`: Environment name, default is `loc`.
- `APP_VERBOSE`: Verbose level, default is 1.
- `APP_CRONTAB`: If a cron scheduler is set, Crontab file name, default is empty.
- `APP_API`: If an API is set, API name, default is empty.
- `APP_QUEUE`: If a queue listener is set, queue name, default is empty.
- `LOC_TIMEZONE`: Local timezone for timestamps, default is `Etc/UTC`.

Default database parameters:

- `DB_ADAPTER`: Database adapter.
- `DB_HOST`: Database host.
- `DB_PORT`: Database port.
- `DB_USERNAME`: Database username.
- `DB_PASSWORD`: Database password.
- `DB_STAGE_DIR`: Database local stage directory, where data is stored before database loading.
- `DB_DATABASE`: Database name.

Supported database adapters include the following:

- `MySQL`
- `PostgreSQL`
- `Redshift`
- `BigQuery` (only as read source)
- `DynamoDB` (only as read source)
- `Mongo` (only as read source)
- `Salesforce` (only as read source)

## Who is ManyThings?

[ManyThings](https://manythings.pro) is a consulting company founded by [Luis Sancho](https://github.com/luissancho) in 2010, specialized in applying data analysis and artificial intelligence solutions to business problems.

Our work involves identifying and solving real-world business issues by leveraging the latest trends in data science, with a fundamentally pragmatic approach aimed at providing visibility, automating, and streamlining complex processes.

Our mission is to assist businesses across all sectors in designing and executing their strategy, as well as leveraging their data to optimize their operations.

We combine the best of data-driven strategy with the most advanced and exclusive technological and business intelligence tools at the service of the client.
