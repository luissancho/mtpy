from typing import Optional
from typing_extensions import Self

from ..data import SQLAdapter


class MySQL(SQLAdapter):

    driver = 'mysql+pymysql'
    default_port = 3306
    quote = '`'
    sessions = False
    data_types = {
        'varchar': 'VARCHAR',
        'text': 'TEXT',
        'date': 'DATE',
        'datetime': 'DATETIME',
        'numeric': 'DECIMAL',
        'tinyint': 'TINYINT',
        'smallint': 'SMALLINT',
        'mediumint': 'MEDIUMINT',
        'integer': 'INTEGER',
        'bigint': 'BIGINT',
        'boolean': 'TINYINT'
    }
    bool_as_int = True
    autokey_suffix = 'AUTO_INCREMENT'
    date_groupings = {
        'year': "CONCAT(YEAR({col}), '-01-01')",
        'month': "STR_TO_DATE(CONCAT(EXTRACT(YEAR_MONTH FROM {col}), '01'), '%Y%m%d')",
        'week': "STR_TO_DATE(CONCAT(YEARWEEK({col}), ' Monday'), '%X%V %W')",
        'day': "DATE({col})"
    }
    on_conflict_suffix = 'ON DUPLICATE KEY UPDATE'
    on_conflict_update = '{col} = VALUES({col})'
    create_relations = 'CONSTRAINT {name} FOREIGN KEY ({key}) REFERENCES {table}'
    create_suffix = 'ENGINE=InnoDB DEFAULT CHARSET=utf8'
    
    def get_version(
        self
    ) -> str:
        return self.get_var(query='SELECT VERSION()')
    
    def get_tables(
        self,
        schema: Optional[str] = None
    ) -> list[str]:
        query = "SHOW TABLES IN {}".format(schema or self.database)
        
        tables = self.get_results(query=query)['Tables_in_{}'.format(schema or self.database)].tolist()

        return tables
    
    def get_columns(
        self,
        table: str
    ) -> list[str]:
        query = "SHOW COLUMNS FROM {}".format(table)
        
        columns = self.get_results(query=query)['Field'].tolist()

        return columns
    
    def get_meta(
        self,
        table: str
    ) -> dict[str, dict]:
        meta = {
            col: {} for col in self.get_columns(table)
        }

        return meta
