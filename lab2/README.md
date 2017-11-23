### Scalable Machine Learning and Deep Learning
#### Lab 2 - Deep Learning with TensorFlow

The getting started notebooks come with the Tensorflow Docker installation: 

https://www.tensorflow.org/versions/r0.12/get_started/os_setup#docker_installation

The template code for the lab comes from here:

https://github.com/id2223/ht17-lab2/blob/master/template.py

To run the notebooks in a Docker container run:

`docker run -p 8888:8888 --name tensorflow-id2223 -v $PATH_TO_HOST_FOLDER:/notebooks -it gcr.io/tensorflow/tensorflow`

where the 'host folder' is your local path to the /lab2 folder.
