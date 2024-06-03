# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 6000 available to the world outside this container
EXPOSE 6000

# Run app.py when the container launches
CMD ["python", "app.py"]



# docker build -t my-flask-app .
# docker run -p 6000:6000 my-flask-app

# docker tag local-image:tagname new-repo:tagname
# docker push new-repo:tagname

# botaniqube
# docker tag my-flask-app antonidabrowski/botaniqube:latest
# docker push antonidabrowski/botaniqube:latest