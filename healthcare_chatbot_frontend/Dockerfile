FROM node:14-alpine

WORKDIR /app

# Copy the package.json and package-lock.json files
COPY package.json .

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the application
RUN npm install

# Expose the ports on which the application will run
EXPOSE 3000

# Start the application on both ports
CMD ["npm", "start"]