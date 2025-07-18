# Cliox-Blockathon25 – Usage Guide

Welcome! This repository is structured to help you get started quickly with your own algorithm implementation for the Blockathon 2025 event.

## Folder Overview

- **sample_timeseries_forecast**  
  This folder contains a **sample project**. It’s provided to give you a basic idea of what your own implementation should look like. You can refer to its structure, code, and setup as an example.

- **template**  
  This is the **template** you should use to create your own solution. It contains all the necessary files and a clear structure, with `TODO` comments to guide your implementation.

Sample adapted from [AgrospAI/ocean-algo – timeseries-forecast](https://github.com/AgrospAI/ocean-algo/tree/main/timeseries-forecast)

## How to Use the Template

1. **Work only in `template/algorithm/src/implementation/algorithm.py`**  
   - This is the **only file you should modify** for implementing your algorithm.
   - Follow the `TODO` comments in this file to implement your algorithm.

2. **Update the `requirment.txt`** to match your needs 

3. **Do not change other files** in the template folder unless specifically instructed.



## Tutorial: Building and Running with Docker Compose

Here’s a step-by-step guide to set up your environment, install dependencies, and run your code using Docker Compose.

### 0. Add sample Data inplace 
* add your sample data to file `template/_data/inputs/eb60f87363a36a5ae5cb8373524a8fd976b0cc5f8c40a706c615b857ae0e2974/0`

### 1. Add Your Dependencies


* Optional 1: 
  - Open `template/algorithm/requirements.txt`.
  - Add any Python packages with version your algorithm needs, one per line.  
    _Example:_
    ```
    numpy==2.2.4
    pandas==2.2.3
    ```

* Optional 2 (recommanded):
  - Navigate to `template/algorithm`
  - update dependencies in the file `pyproject.toml`
  - Use comment below to generate `requirements.txt` file for libraries 
    ``` sh
    uv pip compile pyproject.toml -o requirements.txt 
    ``` 



### 2. Build the Docker Image

Open a terminal and navigate to the `template/` directory:

```sh
cd template
docker compose build
```

This command will:
- Read the `Dockerfile` and `requirements.txt`
- Install your dependencies
- Prepare the environment for running your code

### 3. Run Your Algorithm

After building, you can run your code with:

```sh
docker compose up
```

This will:
- Start the container
- Execute the main script, which will call your implementation in `algorithm.py`

### 4. Check whether result matches your expectation
- Navigate to `template\_data\outputs` to check the algo output

### 5. (Optional) Stopping the Container

To stop the running container, press `Ctrl+C` in the terminal, then run:

```sh
docker compose down
```

## Publishing Your Docker Image

Once your algorithm is working locally, you can publish your Docker image to make it available for deployment or sharing.

> **Prerequisites:** You'll need a Docker Hub account to publish images. If you don't have one, [create a free account at hub.docker.com](https://hub.docker.com/signup).

### 1. Tag Your Image

First, tag your built image with a meaningful name and version:

```sh
# Navigate to the template directory
cd template

# Tag the image (replace 'your-algorithm-name' with your actual name)
docker tag blockathon2025-template-algorithm:latest your-username/your-algorithm-name:latest
docker tag blockathon2025-template-algorithm:latest your-username/your-algorithm-name:v1.0.0
```

### 2. Push to Docker Hub

If you want to publish to Docker Hub:

```sh
# Login to Docker Hub (you'll be prompted for credentials)
docker login

# Push your tagged images
docker push your-username/your-algorithm-name:latest
docker push your-username/your-algorithm-name:v1.0.0
```
