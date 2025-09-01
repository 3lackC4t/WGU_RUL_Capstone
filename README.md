# WGU_RUL_Capstone
## Description
Unexpected bearing failures can result in millions of losses per year in lost production in manufacturing as well as potential interruption of service in critical industries. These failures are most often a result of fatigue failure of rotating bearings subject axial and radial thrust as a result of supporting the load of a piece of rotating machinery. Over time the bearings degrade and become damaged until they reach a material failure point, resulting in the failure of the machine as well. The ability to predict these failures in time to perform preventive maintenance would result in millions of annual revenue protected as well as substantial reductions in service interruptions in critical industries such as power, water treatment, or oil and gas.

This model can be used to predict early failure of rotational bearings and can be used to assist in predictive maintenance of such machines.

## Running The Application
### From Docker
The preffered method of running the bearing RUL predictor is via it's docker file. 

To do this simply clone the repository, build the image, and run the docker container mapping a local port to 5000 for the container.

```sh
docker run -p 5000:5000 <container name>
```

Or, more conveniently, using the docker compose file example below

```yaml
services:
  wgurulcapstone:
    image: wgurulcapstone
    build:
      context: .
      dockerfile: ./Dockerfile
    ports:
      - 5000:5000
```

## Model Training
1. If desired the model can be trained again for demonstration
The model will check for the presence of the autoencoder, reference_data and preprocessor files in the model_data directory when it runs, if it finds them it will load them from save data. 
2. In order to trigger training, simply delete those files and rerun the program as above. 
3. The current proportion setting on the model is 1.0 for full model training. This will take roughly 30 minutes to fully train. If desired, reduce proportion for faster training.
