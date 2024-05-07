from supabase import create_client, Client
import psycopg2
from urllib.parse import urlparse
import pandas as pd
import os
import toml
from dotenv import load_dotenv

load_dotenv()
config_file = toml.load("config.toml")

# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# # supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SUPABASE_TABLE_FEATURE = config_file["supabase"]["table"]["table_features"]

class SupabaseConnect:
    def __init__(self) -> None:
        self.connection_string = "postgres://postgres.sxoqzllwkjfluhskqlfl:5giE*5Y5Uexi3P2@aws-0-us-west-1.pooler.supabase.com:6543/postgres"

    def connect_supabase(self):
        """
        Connects to the Supabase database using the connection string provided.
        
        Returns:
        psycopg2.connection: A connection object to the Supabase database.
        """
        url = urlparse(self.connection_string)
        self.conn = psycopg2.connect(
            database=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port
        )
        return self.conn
    
    def create_cursor(self):
        # Create a cursor
        self.cursor = self.conn.cursor()

    def execute_query(self, table_name, query=None):
        """
        Executes a query on the Supabase database.
        
        Args:
            table_name (str): The name of the table to query
            query (str, optional): The SQL query to execute. Defaults to None.
        """
        self.connect_supabase()
        self.create_cursor()
        if query is None:
            query = f"""
            SELECT *
            FROM {table_name}
            """
        self.cursor.execute(query)
        result = self.cursor.fetchall() 
        self.data  = pd.DataFrame(result, columns=SUPABASE_TABLE_FEATURE) 
        return self.data


