const winston = require('winston');
const LogstashTransport = require('winston-logstash');

// Get the NodeIP and NodePort of the Logstash Service
const LOGSTASH_SERVICE_HOST = '192.168.49.2'; // Replace with the NodeIP of any Node in the cluster
const LOGSTASH_SERVICE_PORT = 30044; // Assuming Logstash Service is exposed on NodePort 30044

const logger = winston.createLogger({
  level: 'info', // Set the desired log level
  transports: [
    new LogstashTransport({
      host: LOGSTASH_SERVICE_HOST,
      port: LOGSTASH_SERVICE_PORT,
      maxPending: 100, // Max number of logs to queue before dropping
      ssl: false, // Set to true if using SSL
    }),
  ],
});

export default logger;

