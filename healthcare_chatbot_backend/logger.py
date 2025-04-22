import logging
from logstash_async.handler import AsynchronousLogstashHandler
from logstash_async.formatter import LogstashFormatter

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get the NodeIP and NodePort of the Logstash Service
LOGSTASH_SERVICE_HOST = '192.168.49.2'  # Replace with the NodeIP of any Node in the cluster
LOGSTASH_SERVICE_PORT = 30044  # Assuming Logstash Service is exposed on NodePort 30044

# Create a Logstash handler (using asynchronous handler for better performance)
logstash_handler = AsynchronousLogstashHandler(
    host=LOGSTASH_SERVICE_HOST,
    port=LOGSTASH_SERVICE_PORT,
    database_path='logstash.db'  # Path to the SQLite database for buffering
)

# Create a formatter for the Logstash handler
formatter = LogstashFormatter()
logstash_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(logstash_handler)

# Log an info message with additional context
logger.info('This is an info message', extra={'app': 'flask-app'})
