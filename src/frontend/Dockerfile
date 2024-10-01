# Dockerfile for the frontend in Astro
FROM node:18-alpine

# Create working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install dependencies
RUN npm install

# Build the Astro application
RUN npm run build

# Expose port 4321
EXPOSE 4321

# Command to start Astro in development mode
CMD ["npm", "run", "start", "--host"]
